import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import onnxruntime as ort
from efficientvit.sam_model_zoo import create_sam_model

medsam_lite_model = create_sam_model("l0", False)
medsam_lite_model.load_state_dict(torch.load("models/CT.pth", map_location='cpu'))
medsam_lite_model.prompt_encoder.input_image_size=(256,256)
medsam_lite_model.to("cpu")
medsam_lite_model.eval()
os.makedirs("onnxmodels", exist_ok=True)
np.save("onnxmodels/positional_encoding.npy", medsam_lite_model.prompt_encoder.get_dense_pe().numpy()) #

models=glob.glob("models/*")

image = torch.randn(1, 3, 256, 256) #req grad?
prompt = {"points":None,"boxes":torch.tensor([[[[100.0,200,100,200]]]]),"masks":None}

for m in models:
    model_name = m.split("/")[-1].split(".pth")[0]
    print(model_name)
    medsam_lite_model.load_state_dict(torch.load(m, map_location='cpu'))
    medsam_lite_model.eval()
    image_embedding = medsam_lite_model.image_encoder(image)
    sparse_embeddings, dense_embeddings = medsam_lite_model.prompt_encoder(**prompt)
    mdin = {'image_embeddings': image_embedding, 'image_pe': medsam_lite_model.prompt_encoder.get_dense_pe(), 'sparse_prompt_embeddings': sparse_embeddings, 'dense_prompt_embeddings': dense_embeddings, 'multimask_output': False}
    low_res_masks, iou_predictions = medsam_lite_model.mask_decoder(**mdin)
    dynamic_axes_img={'image' : {0 : 'batch_size'}, 'image_embedding' : {0 : 'batch_size'}}
    dynamic_axes_prompt={'points' : {0 : 'batch_size'}, 'boxes' : {0 : 'batch_size'}, 'masks' : {0 : 'batch_size'}, 'sparse_embeddings' : {0 : 'batch_size'}, 'dense_embeddings':{0 : 'batch_size'}}
    dynamic_axes_mdin={'image_embeddings' : {0 : 'batch_size'}, 'image_pe' : {0 : 'batch_size'}, 'sparse_prompt_embeddings':{0 : 'batch_size'}, 'dense_prompt_embeddings':{0 : 'batch_size'}, 'multimask_output':{0 : 'batch_size'},'low_res_masks':{0 : 'batch_size'},'iou_predictions':{0 : 'batch_size'}}
    torch.onnx.export(medsam_lite_model.image_encoder,image, "onnxmodels/"+model_name+"_image_encoder.onnx", input_names = ['image'], output_names = ['image_embedding'], dynamic_axes=dynamic_axes_img)
    torch.onnx.export(medsam_lite_model.prompt_encoder,(None, torch.tensor([[[[100.0,200,100,200]]]]), None), "onnxmodels/"+model_name+"_prompt_encoder.onnx", input_names = ['boxes', 'points', 'masks'], output_names = ['sparse_embeddings', 'dense_embeddings'], dynamic_axes=dynamic_axes_prompt) # wrong input_names order, because it only keeps the first input name
    torch.onnx.export(medsam_lite_model.mask_decoder,mdin, "onnxmodels/"+model_name+"_mask_decoder.onnx", input_names = list(mdin.keys()), output_names = ['low_res_masks', 'iou_predictions'])#, dynamic_axes=dynamic_axes_mdin)
    
    ie = ort.InferenceSession("onnxmodels/"+model_name+"_image_encoder.onnx")
    pe = ort.InferenceSession("onnxmodels/"+model_name+"_prompt_encoder.onnx")
    md = ort.InferenceSession("onnxmodels/"+model_name+"_mask_decoder.onnx")
    positional_encoding = np.load("onnxmodels/positional_encoding.npy")
    o_image_embedding = ie.run(["image_embedding"], {"image":image.numpy()})[0]
    o_sparse_embeddings, o_dense_embeddings = pe.run(["sparse_embeddings", "dense_embeddings"], {"boxes":prompt["boxes"].numpy()})
    o_low_res_masks, o_iou_predictions = md.run(["low_res_masks", "iou_predictions"], {"image_embeddings":o_image_embedding, "image_pe": positional_encoding, "sparse_prompt_embeddings": o_sparse_embeddings, "dense_prompt_embeddings": o_dense_embeddings})
    assert (np.allclose(o_image_embedding,image_embedding.detach().numpy(), atol=1e-03)), "onnx and pytorch not close"
    assert (np.allclose(o_sparse_embeddings,sparse_embeddings.detach().numpy(), atol=1e-03)), "onnx and pytorch not close"
    assert (np.allclose(o_dense_embeddings,dense_embeddings.detach().numpy(), atol=1e-03)), "onnx and pytorch not close"
    assert (np.allclose(o_low_res_masks,low_res_masks.detach().numpy(), atol=1e-03)), "onnx and pytorch not close"
    assert (np.allclose(o_iou_predictions,iou_predictions.detach().numpy(), atol=1e-03)), "onnx and pytorch not close"
