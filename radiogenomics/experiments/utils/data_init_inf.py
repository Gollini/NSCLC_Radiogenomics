"""
Author: Ivo Gollini Navarrete
Date: 14/oct/2022
Institution: MBZUAI
"""

#Imports
import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

class DatasetInit(Dataset):
    """
    Dataset class for the msd dataset.
    """
    NAME = "msd"
    IMG_EXT = ".pt"

    # The modes supported by this dataset loader class
    SEGMENTATION_MODES = ["vanilla", "a2t"]

    SUBSETS = ["test"]

    def __init__(
        self,
        path,
        subset="test",
        mode="vanilla",
        channels=1,
    ) -> None:

        if subset not in self.SUBSETS:
            raise ValueError("""Specified subset for dataset is not recognized.""")
        self.subset = subset
        
        # Check that the modes are supported
        if (
            mode not in self.SEGMENTATION_MODES
        ):
            raise ValueError("Unrecognised modes were selected for the Radiogenomics dataset.")

        self.mode = mode
        self.channels = channels
        self.path = path
        self.data = os.listdir(path)
        print("Dataset loaded with {} samples".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        img_path = os.path.join(self.path, patient, patient+'_ct.pt')
        # target_path = os.path.join(self.path, patient, patient+'_seg.pt')

        img = torch.load(img_path)
        # target = torch.load(target_path).astype(np.float32)

        if self.mode == "vanilla":
            # return img , target
            return img, patient

        elif self.mode == "a2t":
            organ_path = os.path.join(self.path, patient, patient+'_bbx.pt')
            organ = torch.load(organ_path).astype(np.float32)
            # return img, organ , target
            return img, organ, patient
        