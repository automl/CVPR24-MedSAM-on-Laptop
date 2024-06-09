import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("weights")
args = parser.parse_args()

medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64)
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

class MedSAM_Lite(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

a=torch.load("../work_dir/LiteMedSAM/lite_medsam.pth", map_location=torch.device('cpu'))
medsam_lite_model.load_state_dict(a)
torch.save(medsam_lite_model.prompt_encoder.state_dict(), "pe.pth")
torch.save(medsam_lite_model.mask_decoder.state_dict(), "md.pth")

from efficientvit.sam_model_zoo import create_sam_model
medsam_lite_model = create_sam_model("l0", False)

medsam_lite_model = medsam_lite_model.image_encoder
medsam_lite_model = nn.DataParallel(medsam_lite_model)
a=torch.load(args.checkpoint, map_location=torch.device('cpu'))["model"]
medsam_lite_model.load_state_dict(a)
torch.save(medsam_lite_model.module.state_dict(), "ie.pth")

medsam_lite_model = create_sam_model("l0", False)
medsam_lite_model.image_encoder.load_state_dict(torch.load("ie.pth", map_location=torch.device('cpu')))
medsam_lite_model.prompt_encoder.load_state_dict(torch.load("pe.pth", map_location=torch.device('cpu')))
medsam_lite_model.mask_decoder.load_state_dict(torch.load("md.pth", map_location=torch.device('cpu')))

torch.save(medsam_lite_model.state_dict(), args.weights)
