import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.backbones.voxelmorph.torch import layers
from models.backbones.voxelmorph.torch.networks import Unet
from models.backbones.voxelmorph.torch.modelio import LoadableModel, store_config_args

class voxelMorphComplex(LoadableModel):

    @store_config_args
    def __init__(self,
        img_size='(128,128,16)',
        nb_unet_features='[[16,32,32,32],[32,32,32,32,32,16,16]]',
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        int_steps='7',
        int_downsize='2',
        src_feats=1,
        trg_feats=1,
        unet_half_res=False
    ):

        super().__init__()

        self.training = True

        self.img_size = eval(img_size)
        self.nb_unet_features = eval(nb_unet_features)
        self.int_steps = int(int_steps)
        self.int_downsize = int(int_downsize)

        print('img_size: %s, nb_unet_features: %s, int_steps: %d, int_downsize: %d' % (self.img_size, self.nb_unet_features, self.int_steps, self.int_downsize))

        ndims = len(self.img_size)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.unet_model = Unet(
            self.img_size,
            infeats=(src_feats + trg_feats),
            nb_features=self.nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if not unet_half_res and self.int_steps > 0 and self.int_downsize > 1:
            self.resize = layers.ResizeTransform(self.int_downsize, ndims)
        else:
            self.resize = None

        if self.int_steps > 0 and self.int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / self.int_downsize, ndims)
        else:
            self.fullsize = None

        down_shape = [int(dim / self.int_downsize) for dim in self.img_size]
        self.integrate = layers.VecInt(down_shape, self.int_steps) if self.int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(self.img_size)

    def forward(self, x, y, x_pts=None, y_pts=None, registration=False):

        source, target = x, y
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        flow_field = self.flow(x)

        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow, pos_flow
        else:
            return y_source, pos_flow