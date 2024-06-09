import glob
import os
import openvino as ov
from os.path import basename
from shutil import copy

models=glob.glob("onnxmodels/*.onnx")
os.makedirs("openvinomodels", exist_ok=True)

for m in models:
    print(m)
    ov.save_model(ov.convert_model(m), "openvinomodels/"+basename(m).replace(".onnx", ".xml"))

positional_encodings=glob.glob("onnxmodels/*.npy")
for pe in positional_encodings:
    print(pe)
    copy(pe, pe.replace("onnxmodels", "openvinomodels"))