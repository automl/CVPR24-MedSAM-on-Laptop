import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import onnxruntime as ort

import openvino as ov
from os.path import basename
from shutil import copy

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT

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

onnxpath="liteonnx"
openvinopath="liteov"

image = torch.randn(1, 3, 256, 256) #req grad?
prompt = {"points":None,"boxes":torch.tensor([[[[100.0,200,100,200]]]]),"masks":None}

medsam_lite_model.load_state_dict(torch.load("work_dir/LiteMedSAM/lite_medsam.pth", map_location='cpu'))
medsam_lite_model.eval()
positional_encoding=medsam_lite_model.prompt_encoder.get_dense_pe().numpy()

os.makedirs(onnxpath, exist_ok=True)

image_embedding = medsam_lite_model.image_encoder(image)
sparse_embeddings, dense_embeddings = medsam_lite_model.prompt_encoder(**prompt)
mdin = {'image_embeddings': image_embedding, 'image_pe': medsam_lite_model.prompt_encoder.get_dense_pe(), 'sparse_prompt_embeddings': sparse_embeddings, 'dense_prompt_embeddings': dense_embeddings, 'multimask_output': False}
low_res_masks, iou_predictions = medsam_lite_model.mask_decoder(**mdin)
dynamic_axes_img={'image' : {0 : 'batch_size'}, 'image_embedding' : {0 : 'batch_size'}}
dynamic_axes_prompt={'points' : {0 : 'batch_size'}, 'boxes' : {0 : 'batch_size'}, 'masks' : {0 : 'batch_size'}, 'sparse_embeddings' : {0 : 'batch_size'}, 'dense_embeddings':{0 : 'batch_size'}}
dynamic_axes_mdin={'image_embeddings' : {0 : 'batch_size'}, 'image_pe' : {0 : 'batch_size'}, 'sparse_prompt_embeddings':{0 : 'batch_size'}, 'dense_prompt_embeddings':{0 : 'batch_size'}, 'multimask_output':{0 : 'batch_size'},'low_res_masks':{0 : 'batch_size'},'iou_predictions':{0 : 'batch_size'}}
torch.onnx.export(medsam_lite_model.image_encoder,image, onnxpath+"/lite_medsam_image_encoder.onnx", input_names = ['image'], output_names = ['image_embedding'], dynamic_axes=dynamic_axes_img)
torch.onnx.export(medsam_lite_model.prompt_encoder,(None, torch.tensor([[[[100.0,200,100,200]]]]), None), onnxpath+"/lite_medsam_prompt_encoder.onnx", input_names = ['boxes', 'points', 'masks'], output_names = ['sparse_embeddings', 'dense_embeddings'], dynamic_axes=dynamic_axes_prompt) # wrong input_names order, because it only keeps the first input name
torch.onnx.export(medsam_lite_model.mask_decoder,mdin, onnxpath+"/lite_medsam_mask_decoder.onnx", input_names = list(mdin.keys()), output_names = ['low_res_masks', 'iou_predictions'])#, dynamic_axes=dynamic_axes_mdin)

ie = ort.InferenceSession(onnxpath+"/lite_medsam_image_encoder.onnx")
pe = ort.InferenceSession(onnxpath+"/lite_medsam_prompt_encoder.onnx")
md = ort.InferenceSession(onnxpath+"/lite_medsam_mask_decoder.onnx")
o_image_embedding = ie.run(["image_embedding"], {"image":image.numpy()})[0]
o_sparse_embeddings, o_dense_embeddings = pe.run(["sparse_embeddings", "dense_embeddings"], {"boxes":prompt["boxes"].numpy()})
o_low_res_masks, o_iou_predictions = md.run(["low_res_masks", "iou_predictions"], {"image_embeddings":o_image_embedding, "image_pe": positional_encoding, "sparse_prompt_embeddings": o_sparse_embeddings, "dense_prompt_embeddings": o_dense_embeddings})
assert (np.allclose(o_image_embedding,image_embedding.detach().numpy(), atol=1e-04)), "onnx and pytorch not close"
assert (np.allclose(o_sparse_embeddings,sparse_embeddings.detach().numpy(), atol=1e-04)), "onnx and pytorch not close"
assert (np.allclose(o_dense_embeddings,dense_embeddings.detach().numpy(), atol=1e-04)), "onnx and pytorch not close"
assert (np.allclose(o_low_res_masks,low_res_masks.detach().numpy(), atol=1e-04)), "onnx and pytorch not close"
assert (np.allclose(o_iou_predictions,iou_predictions.detach().numpy(), atol=1e-04)), "onnx and pytorch not close"

os.makedirs(openvinopath, exist_ok=True)
for m in glob.glob(onnxpath+"/*.onnx"):
    ov.save_model(ov.convert_model(m), openvinopath+"/"+basename(m).replace(".onnx", ".xml"))
