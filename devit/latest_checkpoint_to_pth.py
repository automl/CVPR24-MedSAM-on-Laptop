import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from efficientvit.sam_model_zoo import create_sam_model
medsam_lite_model = create_sam_model("l0", False)
medsam_lite_model.prompt_encoder.input_image_size=(256,256)
medsam_lite_model=nn.DataParallel(medsam_lite_model)

a=torch.load("medsam_lite_latest.pth", map_location=torch.device('cpu'))["model"]
medsam_lite_model.load_state_dict(a)
torch.save(medsam_lite_model.module.state_dict(), "devit_24_50.pth")
