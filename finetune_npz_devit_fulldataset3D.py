# only works when all data subsets are 3D
# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from efficientvit.sam_model_zoo import create_sam_model
import cv2
import torch.nn.functional as F
import pandas as pd
from multiprocessing import pool

from matplotlib import pyplot as plt
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default="train_npz",
    help="Path to the npy data root."
)
parser.add_argument(
    "-data_subset", type=str, default="",
    help="prefix for data subset, multiple ones separated by |"
)
parser.add_argument(
    "-pretrained_checkpoint", type=str, default="lite_medsam.pth",
    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
    "-resume", type=str, default='work_dir_evit/base/medsam_lite_latest.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="work_dir_evit/base",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=10,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=16,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=0.00005,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)

args = parser.parse_args()
# %%
work_dir = args.work_dir
data_root = args.data_root
data_subset = args.data_subset
medsam_lite_checkpoint = args.pretrained_checkpoint
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
do_sancheck = args.sanity_check
checkpoint = args.resume

makedirs(work_dir, exist_ok=True)

# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def cal_iou(result, reference):

    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])

    iou = intersection.float() / union.float()

    return iou.unsqueeze(1)

def file_to_slices(a):
    i,f=a
    D,_,_=np.load(f)["imgs"].shape
    return [(i,s) for s in range(D)]


# %%
class NpzDataset(Dataset):
    def __init__(self, data_root, filter_csv, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        files=[]
        df=pd.read_csv(filter_csv)
        for pattern in data_subset.split('|'):
            curr_files = [join(data_root,e) for e in df["filename"] if e.startswith(pattern)]
            print(len(curr_files), "files found for pattern "+pattern)
            files.extend(curr_files)
        print(len(files), "files in total")
        self.file_paths = sorted(files)

        
        index_to_slice=[]
        file_list=[(i,f) for i,f in enumerate(self.file_paths)]
        with pool.Pool() as p:
            for slices in tqdm(p.imap(file_to_slices,file_list)):
                index_to_slice.extend(slices)
        

#        index=0
#        file_index=0
#        index_to_slice=[]
#        for f in tqdm(self.file_paths):
#            D,_,_=np.load(f)["imgs"].shape
#            for s in range(D):
#                index_to_slice.append((file_index,s))
#                index+=1
#            file_index+=1
        self.index_to_slice = {k:v for k,v in enumerate(index_to_slice)}


        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return max(self.index_to_slice.keys())

    def __getitem__(self, index):
        file_index, slice_index = self.index_to_slice[index]
        npz = np.load(self.file_paths[file_index], 'r', allow_pickle=True)
        gts = npz["gts"] # multiple labels [0, 1,4,5...], (256,256)
        imgs = npz["imgs"]

        if len(gts.shape) > 2: ## 3D image
            i=slice_index
            img_i = imgs[i, :, :]
            gt_i = gts[i, :, :]
            img_i = self.resize_longest_side(img_i)
            gt_i = cv2.resize(gt_i, (img_i.shape[1], img_i.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_i = self.pad_image(img_i)
            gts = self.pad_image(gt_i)
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)# (H, W, 3)
        else:
            if len(imgs.shape) < 3:
                img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
            else:
                img_3c = imgs
            img_3c = self.resize_longest_side(img_3c)
            gts = cv2.resize(gts, (img_3c.shape[1], img_3c.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_3c = self.pad_image(img_3c)
            gts = self.pad_image(gts)
        gts = np.uint16(gts)


        img_resize = img_3c#self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = img_resize#self.pad_image(img_resize) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        label_ids = np.unique(gts)[1:]
        try:
            gt2D = np.uint8(gts == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(self.file_paths[file_index], 'label_ids.tolist()', label_ids.tolist())
            return self.__getitem__(random.randint(0,len(self)-1))
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        #print(gt2D.shape)
        gt2D=cv2.resize(gt2D,(256,256))#[None, :,:]
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": self.file_paths[file_index],
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded

medsam_lite_model = create_sam_model("l0", False)
medsam_lite_model.prompt_encoder.input_image_size=(256,256)
medsam_lite_model.to(device)
medsam_lite_model.eval()

if medsam_lite_checkpoint is not None:
    if isfile(medsam_lite_checkpoint):
        print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
        medsam_lite_ckpt = torch.load(
            medsam_lite_checkpoint,
            map_location="cpu"
        )
        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
    else:
        print(f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch")

medsam_lite_model = nn.DataParallel(medsam_lite_model)
medsam_lite_model = medsam_lite_model.to(device)
medsam_lite_model.train()

# %%
print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
# %%
optimizer = optim.AdamW(
    medsam_lite_model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
iou_loss = nn.MSELoss(reduction='mean')
# %%
train_dataset = NpzDataset(data_root=data_root,filter_csv="fulldataset.csv", data_aug=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

if checkpoint and isfile(checkpoint):
    print(f"Resuming from checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint)
    medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = 1e10
# %%
for parameter in medsam_lite_model.module.prompt_encoder.parameters():
    parameter.requires_grad = False

train_losses = []
for epoch in range(start_epoch + 1, num_epochs+1):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    medsam_lite_model.train()
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        boxes = batch["bboxes"]
        optimizer.zero_grad()
        image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)

        image_embedding = medsam_lite_model.module.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = medsam_lite_model.module.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        logits_pred, iou_pred = medsam_lite_model.module.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=medsam_lite_model.module.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        ) # (B, 1, 256, 256)
        #print(logits_pred.shape)

        #logits_pred, iou_pred = medsam_lite_model(image, boxes)
        l_seg = seg_loss(logits_pred, gt2D)
        l_ce = ce_loss(logits_pred, gt2D.float())
        #mask_loss = l_seg + l_ce
        mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
        iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = iou_loss(iou_pred, iou_gt)
        #loss = mask_loss + l_iou
        loss = mask_loss + iou_loss_weight * l_iou
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_end_time = time()
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    train_losses.append(epoch_loss_reduced)
    lr_scheduler.step(epoch_loss_reduced)
    model_weights = medsam_lite_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": epoch_loss_reduced if epoch_loss_reduced < best_loss else best_loss,
    }
    torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss in epoch {epoch}: {best_loss:.5f} -> {epoch_loss_reduced:.5f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))

    # %% plot loss
    plt.plot(train_losses)
    plt.title("Dice + Binary Cross Entropy + IoU Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(work_dir, "train_loss.png"))
    plt.close()
