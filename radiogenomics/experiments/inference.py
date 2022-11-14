"""
Author: Ivo Gollini Navarrete
Date: 14/nov/2022
Institution: MBZUAI
"""

#Imports
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.utils import data_init_inf

from models.unet import UNet
from models.ra_seg import RA_Seg

from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric


MODELS ={
    "UNet": UNet,
    "RA_Seg": RA_Seg
}

CRITERIONS = {
    "CE": nn.CrossEntropyLoss,
    "BCE": nn.BCELoss,
    "BCEL": nn.BCEWithLogitsLoss,
    "FOCAL": FocalLoss,
    "DICE": DiceLoss,
    "DICE_CE": DiceCELoss,
}

METRIC = {
    "DICE": DiceMetric,
}

"""Class to perform inference on a trained model"""
class Inference:
    def __init__(self, data_path, weights_path, model_class):
        self.data_path = data_path
        self.weights_path = weights_path
        self.model_class= model_class

        # Set GPU
        self.device = torch.device(
            "cuda:" + str(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        )
        print("Running on GPU:", torch.cuda.current_device())

        # Initialize model
        self.model = MODELS[model_class](
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=[64, 128, 256, 512, 512],
            strides=[2, 2, 2, 2],
            ).to(self.device)
        print("Model {} loaded".format(model_class))

        # Load weights
        self.model.load_state_dict(torch.load(self.weights_path))
        print("Weights loaded from {}".format(self.weights_path))

        # Initizalize dataset
        self.testset = data_init_inf.DatasetInit(
                path = self.data_path,
                subset="test",
                channels=[64, 128, 256, 512, 512],
                mode = "a2t",
            )

        self.test_loader = DataLoader(
                self.testset,
                batch_size=1,
                num_workers=4,
                shuffle=False
            )

        self.criterion = CRITERIONS["DICE_CE"](to_onehot_y=True)
        self.metric = METRIC["DICE"](include_background=False, reduction="mean")

    def post_process(self, img):
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        img = img.to(torch.int64)
        return img

    def feat_preprocess(self, features, patient):
        vector = features.mean(dim=(-2,-1)).flatten(start_dim=1)
        vector = vector.squeeze().cpu().detach().numpy()
        print(vector.shape)
        np.save(os.path.join(self.data_path, patient, "{}_raw_feat.npy".format(patient)), vector)
        print('Raw high-Level features extracted.')
        return

    def save_seg(self, seg, patient):
        seg = seg.squeeze().cpu().detach().numpy()
        np.save(os.path.join(self.data_path, patient, "{}_seg.npy".format(patient)), seg)
        return

    def run(self):
        self.model.eval()

        print("Starting inference...")
        with torch.no_grad():
            for batch_num, data in enumerate(self.test_loader):
                inp = data[0].to(self.device)

                if self.model_class == "RA_Seg":
                    organ = data[1].to(self.device)
                    test_output, hl_feat = self.model(inp, organ) # Tumor segmentation and high level features from encoder2.
                    test_output = self.post_process(test_output) #Binarize
                    self.save_seg(test_output, data[2][0]) # Save segmentation
                    self.feat_preprocess(hl_feat, data[2][0]) #Extract features
