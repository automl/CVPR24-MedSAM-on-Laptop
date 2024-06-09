import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("weights")
args = parser.parse_args()

from efficientvit.sam_model_zoo import create_sam_model
medsam_lite_model = create_sam_model("l0", False)
medsam_lite_model.prompt_encoder.input_image_size=(256,256)
medsam_lite_model=nn.DataParallel(medsam_lite_model)

a=torch.load(args.checkpoint, map_location=torch.device('cpu'))["model"]
medsam_lite_model.load_state_dict(a)
torch.save(medsam_lite_model.module.state_dict(), args.weights)
