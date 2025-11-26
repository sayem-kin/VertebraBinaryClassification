import os
import yaml
import numpy as np
import nibabel as nib
from typing import List
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T


def window_hu(img, WL=-300, WW=800):
   
    lower = WL - WW // 2
    upper = WL + WW // 2

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img.astype(np.float32)

def load_yaml_list(path: str) -> List[str]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def binary_label_from_filename(filename: str) -> int:
    """
    Filename format:
        sub-verseXXX_YY_G.nii.gz
        where G = grade (0, 1, 2, 3)
    Returns:
        0 if grade == 0
        1 if grade in {1,2,3}
    """
    grade = int(filename.split("_")[-1].replace(".nii.gz", ""))
    return 0 if grade == 0 else 1

class VertebraSagittalDataset(Dataset):
    def __init__(self, img_root: str, file_list: List[str],
                 resize: int = 224, augment: bool = False):
        """
        Args:
            img_root (str): folder where vertebra .nii.gz files live
            file_list (List[str]): list of filenames from YAML
            resize (int): final output resolution (224 recommended)
            augment (bool): apply data augmentation (True for train only)
        """
        self.img_root = Path(img_root)
        self.file_list = file_list
        self.resize = resize
        self.augment = augment

        # Simple geometric augmentation on tensors (C, H, W)
        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        full_path = self.img_root / filename

        vol = nib.load(str(full_path)).get_fdata()

        mid = vol.shape[0] // 2
        sag = vol[mid, :, :]   # (H, W)

        
        sag = window_hu(sag)   # np.float32 in [0,1]

        sag = torch.from_numpy(sag).unsqueeze(0)   # (1, H, W)

        if self.transform is not None:
            sag = self.transform(sag)

        sag = F.interpolate(
            sag.unsqueeze(0),      # (1, 1, H, W)
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)              # (1, 224, 224)
        
        label = binary_label_from_filename(filename)
        label = torch.tensor(label, dtype=torch.long)

        return sag, label, filename
