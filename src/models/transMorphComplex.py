import torch
import torch.nn as nn

from models.backbones.transmorph.transMorphCardiac import CONFIGS, TransMorph
from models.backbones.voxelmorph.torch import layers

class transMorphComplex(nn.Module):

    def __init__(self,
        img_size='(128,128,16)',
        int_steps='7',
        int_downsize='2',
        trans_type='l',
    ):
        super().__init__()

        self.training = True

        self.img_size = eval(img_size)
        self.int_steps = int(int_steps)
        self.int_downsize = int(int_downsize)
        self.trans_type = trans_type

        print("img_size: {}, int_steps: {}, int_downsize: {}, trans_type".format(self.img_size, self.int_steps, self.int_downsize, self.trans_type))

        if self.trans_type == 'l':
            self.model = TransMorph(CONFIGS['TransMorph-Large'])
        elif self.trans_type == 's':
            self.model = TransMorph(CONFIGS['TransMorph-Small'])
        elif self.trans_type == 't':
            self.model = TransMorph(CONFIGS['TransMorph-Tiny'])
        elif self.trans_type == 'n':
            self.model = TransMorph(CONFIGS['TransMorph'])

        if self.int_steps > 0 and self.int_downsize > 1:
            self.resize = layers.ResizeTransform(self.int_downsize, 3)
            self.fullsize = layers.ResizeTransform(1 / self.int_downsize, 3)
        else:
            self.resize = None
            self.fullsize = None

        down_shape = [int(dim / self.int_downsize) for dim in self.img_size]
        self.integrate = layers.VecInt(down_shape, self.int_steps) if self.int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(self.img_size)

    def forward(self, source, target, registration=False):

        x = torch.cat([source, target], dim=1)
        b,c,h,w,d = x.shape
        zero_x = torch.zeros((b,c,h,w,d//2)).to(x.device)
        x = torch.cat([zero_x, x, zero_x], dim=-1) # b,c,h,w,2d

        flow_field = self.model(x)
        pos_flow = flow_field[:,:,:,:,8:24]

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