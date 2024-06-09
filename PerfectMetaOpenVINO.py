from os import makedirs
from os.path import join, basename
from glob import glob
#from time import time
import numpy as np

import cv2
import argparse
#from datetime import datetime
import openvino as ov
import openvino.properties as props

core = ov.Core()

#%% set seeds
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='test_demo/imgs/',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='test_demo/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="work_dir/LiteMedSAM/lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay',
    help='directory to save the overlay image'
)
parser.add_argument(
    '--nopandas',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--indocker',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)

args = parser.parse_args()

if args.indocker:
    core.set_property({props.cache_dir: "/workspace/inputs/openvinocache"})
else:
    core.set_property({props.cache_dir: "openvinocache"})

data_root = args.input_dir
pred_save_dir = args.output_dir

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
image_size = 256

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded


def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def medsam_inference(prompt_encoder, mask_decoder, positional_encoding, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    out = prompt_encoder({"boxes":box_256[None, None, ...].astype(np.float32)})
    sparse_embeddings, dense_embeddings = out["sparse_embeddings"], out["dense_embeddings"]
    out = mask_decoder({"image_embeddings":img_embed, "image_pe": positional_encoding, "sparse_prompt_embeddings": sparse_embeddings, "dense_prompt_embeddings": dense_embeddings})
    low_res_logits, iou = out["low_res_masks"], out["iou_predictions"]
    low_res_logits = low_res_logits[..., :new_size[0], :new_size[1]]
    # Resize
    low_res_logits = low_res_logits.squeeze()
    low_res_logits = cv2.resize(low_res_logits, original_size[::-1], interpolation=cv2.INTER_LINEAR)
    medsam_seg = (low_res_logits > 0).astype(np.uint8)
    return medsam_seg, iou

# could also preload all models, but this would make the runtime worse in CVPR24_time_eval.py
pos_encoding = np.load("openvinomodels/positional_encoding.npy")
pe = core.compile_model(model="openvinomodels/lite_medsam_prompt_encoder.xml", device_name="CPU")
sessions = dict()
def load_session(name):
    if name not in sessions:
        ie = core.compile_model(model="openvinomodels/"+name+"_image_encoder.xml", device_name="CPU")
        md = core.compile_model(model="openvinomodels/"+name+"_mask_decoder.xml", device_name="CPU")
        sessions[name] = [ie, pe, md, pos_encoding]
    return sessions[name]

def filename_to_modelname(filename):
    if filename.startswith("3DBox_PET"): return "3D"
    if filename.startswith("3DBox_MR"): return "3D"
    if filename.startswith("3DBox_CT"): return "CT"

    if filename.startswith("2DBox_X-Ray"): return "XRay"
    if filename.startswith("2DBox_XRay"): return "XRay"
    if filename.startswith("2DBox_CXR"): return "XRay"
    if filename.startswith("2DBox_XR"): return "XRay"
    if filename.startswith("2DBox_US"): return "lite_medsam"#"US"
    if filename.startswith("2DBox_Ultra"): return "lite_medsam"#"US"
    if filename.startswith("2DBox_Fundus"): return "Fundus"
    if filename.startswith("2DBox_Endoscopy"): return "Endoscopy"
    if filename.startswith("2DBox_Endoscope"): return "Endoscopy"
    
    if filename.startswith("2DBox_Dermoscope"): return "Dermoscopy"
    if filename.startswith("2DBox_Dermoscopy"): return "Dermoscopy"
    
    if filename.startswith("2DBox_Microscope"): return "Microscopy"
    if filename.startswith("2DBox_Microscopy"): return "Microscopy"

    if filename.startswith("2DBox_CT"): return "CT"
    if filename.startswith("2DBox_MR"): return "3D"
    if filename.startswith("2DBox_PET"): return "3D"

    if filename.startswith("2DBox_Mamm"): return "Mammography"
    if filename.startswith("2DBox_OCT"): return "OCT"

    if "Microscope" in filename: return "Microscopy"
    if "Microscopy" in filename: return "Microscopy"
    if "Dermoscopy" in filename: return "Dermoscopy"
    if "Endoscopy" in filename: return "Endoscopy"
    if "Fundus" in filename: return "Fundus"
    if "X-Ray" in filename: return "XRay"
    if "XRay" in filename: return "XRay"
    if "PET" in filename: return "3D"
    if "OCT" in filename: return "OCT" #makesure OCT stays before CT check
    if "MR" in filename: return "3D"
    if "Mamm" in filename: return "Mammography"
    if "US" in filename: return "lite_medsam"#"US"
    if "CT" in filename: return "CT"
    print(filename, "no match found")
    return "lite_medsam"

def MedSAM_infer_npz_2D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)

    image_encoder, prompt_encoder, mask_decoder, positional_encoding = load_session(filename_to_modelname(npz_name))
    
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint16)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256 = img_256_padded.astype(np.float32).transpose((2, 0, 1))[None, ...]
    image_embedding = image_encoder({"image":img_256})[0]

    for idx, box in enumerate(boxes, start=1):
        x_min, y_min, x_max, y_max = box
        box_mask = np.zeros((H,W), dtype=bool)
        box_mask[x_min:x_max+1,y_min:y_max+1] = True
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(prompt_encoder, mask_decoder, positional_encoding, image_embedding, box256, (newh, neww), (H, W))
        segs[medsam_mask>0 & box_mask] = idx

    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

def MedSAM_infer_npz_3D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)

    image_encoder, prompt_encoder, mask_decoder, positional_encoding = load_session(filename_to_modelname(npz_name))

    def compute_embedding(img_2d):
        if len(img_2d.shape) == 2:
            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
        else:
            img_3c = img_2d
        img_256 = resize_longest_side(img_3c, 256)
        new_H, new_W = img_256.shape[:2]
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        ## Pad image to 256x256
        img_256 = pad_image(img_256)
        # convert the shape to (3, H, W)
        img_256 = img_256.astype(np.float32).transpose((2, 0, 1))[None, ...]
        # get the image embedding
        image_embedding = image_encoder({"image":img_256})[0]
        return  image_embedding, new_H, new_W

    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint16)
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    D, H, W = img_3D.shape
    new_H, new_W = None, None

    image_encoder_cache = dict()
    lookups_left = dict()

    for box3D in boxes_3D:
        _, _, z_min, _, _, z_max = box3D
        z_min = max(0, z_min)
        z_max = min(z_max, D)
        for z in range(z_min, z_max):
            lookups_left[z]=lookups_left.get(z,0)+1

    z_indices = lookups_left.keys()

    image_slices = [img_3D[z, :, :] for z in z_indices]
    image_embeddings = list(map(compute_embedding, image_slices))
    new_H = image_embeddings[0][1]
    new_W = image_embeddings[0][2]
    image_embeddings = [e[0] for e in image_embeddings]
    for z, embedding in zip(z_indices, image_embeddings):
        image_encoder_cache[z] = embedding


    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        z_max = min(z_max,D)
        for z in range(z_middle, z_max):
            image_embedding=image_encoder_cache[z]
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    if np.max(pre_seg256) > 0:
                        box_256 = get_bbox256(pre_seg256)
                    else:
                        box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(prompt_encoder, mask_decoder, positional_encoding, image_embedding, box_256[None, ...], [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        z_min = max(-1, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            image_embedding=image_encoder_cache[z]
            pre_seg = segs_3d_temp[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                if np.max(pre_seg256) > 0:
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(prompt_encoder, mask_decoder, positional_encoding, image_embedding, box_256[None, ...], [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx

        segs[segs_3d_temp>0] = idx

    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )


if __name__ == '__main__':
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    for img_npz_file in img_npz_files:
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(img_npz_file)
        else:
            MedSAM_infer_npz_2D(img_npz_file)

