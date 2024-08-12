import glob
import random
import os
import pandas as pd
import argparse
from pathlib import Path

random.seed(2024)
os.makedirs("datasplit", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("datapath")
args = parser.parse_args()
modalities=glob.glob((Path(args.datapath)/"CVPR24-MedSAMLaptopData/train_npz/*").absolute().as_posix())
train_files=[]
val_files=[]
for m in modalities:
    if os.path.isfile(m):continue
    print(m)
    for subdir in glob.glob(m+"/*"):
        if os.path.isfile(subdir):continue
        npzs = glob.glob(subdir+"/*.npz")
        random.shuffle(npzs)
        l=len(npzs)
        split_index = int(l*0.8)
        train_files.extend(npzs[:split_index])
        val_files.extend(npzs[split_index:])
pd.DataFrame(train_files, columns=['file']).to_csv('datasplit/train.csv', index=False)
pd.DataFrame(val_files, columns=['file']).to_csv('datasplit/val.csv', index=False)
pd.DataFrame(train_files+val_files, columns=['file']).to_csv('datasplit/fulldataset.csv', index=False)
