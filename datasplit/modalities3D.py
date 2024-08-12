import os
import pandas as pd
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("trainpath")
parser.add_argument("valpath")
args = parser.parse_args()
trainfiles=pd.read_csv(args.trainpath)
valfiles=pd.read_csv(args.valpath)

modalities={
    "Dermoscopy":r"^Dermoscopy",
    "Endoscopy":r"^Endoscopy",
    "Fundus":r"^Fundus",
    "Mammography":r"^Mamm",
    "Microscopy":r"^Microscopy",
    "OCT":r"^OCT",
    "US":r"^US",
    "XRay":r"^XRay",
    "3D":r"^(CT|MR|PET)"
}

os.makedirs("modalities3D", exist_ok=True)

for modality, prefix_regex in modalities.items():
    pd.DataFrame(trainfiles[trainfiles['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'modalities3D/{modality}.train.csv', index=False)
    pd.DataFrame(valfiles[valfiles['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'modalities3D/{modality}.val.csv', index=False)

