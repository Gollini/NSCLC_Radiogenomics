"""
Author: Ivo Gollini Navarrete
Date: 12/sep/2022
Institution: MBZUAI

"Attention to recurrence" (A2R) is a method to improve the performance introducing 
attention module to the skip connection to help the model focus on the ROI and
recurrent module at the bottom of the UNET architecture to  capture interslice continuity.
This method is based on:
- "A Teacher-Student Framework for Semi-supervised Medical Image Segmentation From Mixed Supervision".
- "Lung Cancer Tumor Region Segmentation Using Recurrent 3D-DenseUNet".

"""

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
from models.utils.convlstm import ConvLSTM


__all__ = ["ra_seg", "RA_Seg"]


@export("monai.networks.nets")
@alias("Unet")
class RA_Seg(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        act2: Union[Tuple, str] = Act.SIGMOID,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        norm2: Union[Tuple, str] = Norm.BATCH,
        dropout: float = 0.1,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.act2 = act2
        self.norm = norm
        self.norm2 = norm2
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.encoder1 = self._get_down_layer(self.in_channels, self.channels[0], self.strides[0], is_top=True) # (1, 64, 2)
        self.a2o_encoder1 = self._a2o_conv(self.in_channels, self.channels[0], 1)

        self.encoder2 = self._get_down_layer(self.channels[0], self.channels[1], self.strides[1], is_top=False) # (64, 128, 2)
        self.a2o_encoder2 = self._a2o_conv(self.in_channels, self.channels[1], 1)

        self.encoder3 = self._get_down_layer(self.channels[1], self.channels[2], self.strides[2], is_top=False) # (128, 256, 2)
        self.a2o_encoder3 = self._a2o_conv(self.in_channels, self.channels[2], 1)

        self.encoder4 = self._get_down_layer(self.channels[2], self.channels[3], self.strides[3], is_top=False) # (256, 512, 2)
        self.a2o_encoder4 = self._a2o_conv(self.in_channels, self.channels[3], 1)

        self.bottom = self._get_bottom_layer(self.channels[3], self.channels[4]) # (512, 1024), stride = 1

        self.decoder4 = self._get_up_layer((self.channels[4]+self.channels[3]), self.channels[2], self.strides[3], is_top=False) # (1024+512, 256, 2)

        self.decoder3 = self._get_up_layer(self.channels[2]*2, self.channels[1], self.strides[2], is_top=False) # (512, 128, 2)

        self.decoder2 = self._get_up_layer(self.channels[1]*2, self.channels[0], self.strides[1], is_top=False) # (256, 64, 2)

        self.decoder1 = self._get_up_layer(self.channels[0]*2, self.out_channels, self.strides[0], is_top=True) # (128, 1, 2) -> output
        self.activation = nn.Sigmoid()

    def _a2o_conv(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        mod: nn.Module
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act2,
            norm=self.norm2,
            dropout = None,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the Recurrent network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        mod: nn.Module

        mod = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=[out_channels, out_channels, out_channels],
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        return mod

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor, organ: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        skip1 = torch.add(enc1, torch.mul(
            enc1, 
            self.a2o_encoder1(F.interpolate(organ, scale_factor=0.5, mode='trilinear'))
            ))

        enc2 = self.encoder2(enc1)
        skip2 = torch.add(enc2, torch.mul(
            enc2,
            self.a2o_encoder2(F.interpolate(organ, scale_factor=0.25, mode='trilinear'))
            ))

        enc3 = self.encoder3(enc2)
        skip3 = torch.add(enc3, torch.mul(
            enc3,
            self.a2o_encoder3(F.interpolate(organ, scale_factor=0.125, mode='trilinear'))
            ))

        enc4 = self.encoder4(enc3)
        skip4 = torch.add(enc4, torch.mul(
            enc4,
            self.a2o_encoder4(F.interpolate(organ, scale_factor=0.0625, mode='trilinear'))
            ))
        
        """
        Permute from (batch, features, depth, h, w) to (batch, depth, features, h, w) [B, Time-steps, channels, H, W]
        last_state_list = return[0][0]
        """
        bottom = self.bottom(enc4.permute(0, 2, 1, 3, 4))[0][0]

        dec4 = self.decoder4(torch.cat((bottom.permute(0, 2, 1, 3, 4), skip4), dim=1))

        dec3 = self.decoder3(torch.cat((dec4, skip3), dim=1))

        dec2 = self.decoder2(torch.cat((dec3, skip2), dim=1))

        dec1 = self.decoder1(torch.cat((dec2, skip1), dim=1))
        
        return self.activation(dec1), dec2

ra_seg = RA_Seg
