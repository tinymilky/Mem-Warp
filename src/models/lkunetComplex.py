import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from models.backbones.voxelmorph.torch import layers

class LK_encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(LK_encoder, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 3, 1, 1),
            nn.BatchNorm3d(out_cs),
        )
        self.large = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
            nn.BatchNorm3d(out_cs),
        )
        self.one = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 1, 1, 0),
            nn.BatchNorm3d(out_cs),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        if self.in_cs == self.out_cs and self.stride == 1:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3

        return self.prelu(x)

class encoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=3, stride=1, padding=1):
        super(encoder, self).__init__()
        if kernel_size == 3:
            self.layer = nn.Sequential(
                nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
                nn.BatchNorm3d(out_cs),
                nn.PReLU()
            )
        elif kernel_size > 3:
            self.layer = LK_encoder(in_cs, out_cs, kernel_size, stride, padding)

    def forward(self, x):
        return self.layer(x)

class decoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_cs, out_cs, kernel_size, stride, padding, output_padding),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class lkunetComplex(nn.Module):

    def __init__(self,
            img_size='(128,128,16)',
            start_channel = '32',  # N_s in the paper
            lk_size = '5',         # kernel size of LK encoder
            int_steps = '7',       # number of integration steps
            int_downsize = '2',    # downsize factor for integration
        ):

        super(lkunetComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.img_size = eval(img_size)
        self.int_steps = int(int_steps)
        self.int_downsize = int(int_downsize)

        print("img_size: {}, start_channel: {}, lk_size: {}, int_steps: {}, int_downsize: {}".format(self.img_size, self.start_channel, self.lk_size, self.int_steps, self.int_downsize))

        N_s = self.start_channel
        K_s = self.lk_size

        self.flow = nn.Conv3d(N_s*2, 3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.eninput = encoder(2, N_s)
        self.ec1 = encoder(N_s, N_s)
        self.ec2 = encoder(N_s, N_s * 2) # stride=2
        self.ec3 = LK_encoder(N_s * 2, N_s * 2, K_s, 1, K_s//2) # LK encoder
        self.ec4 = encoder(N_s * 2, N_s * 4) # stride=2
        self.ec5 = LK_encoder(N_s * 4, N_s * 4, K_s, 1, K_s//2) # LK encoder
        self.ec6 = encoder(N_s * 4, N_s * 8) # stride=2
        self.ec7 = LK_encoder(N_s * 8, N_s * 8, K_s, 1, K_s//2) # LK encoder
        self.ec8 = encoder(N_s * 8, N_s * 8) # stride=2
        self.ec9 = LK_encoder(N_s * 8, N_s * 8, K_s, 1, K_s//2) # LK encoder

        self.dc1 = encoder(N_s * 16, N_s * 8)
        self.dc2 = encoder(N_s * 8,  N_s * 4)
        self.dc3 = encoder(N_s * 8,  N_s * 4)
        self.dc4 = encoder(N_s * 4,  N_s * 2)
        self.dc5 = encoder(N_s * 4,  N_s * 4)
        self.dc6 = encoder(N_s * 4,  N_s * 2)
        self.dc7 = encoder(N_s * 3,  N_s * 2)
        self.dc8 = encoder(N_s * 2,  N_s * 2)

        self.up1 = decoder(N_s * 8, N_s * 8)
        self.up2 = decoder(N_s * 4, N_s * 4)
        self.up3 = decoder(N_s * 2, N_s * 2)
        self.up4 = decoder(N_s * 2, N_s * 2)

        if self.int_steps > 0 and self.int_downsize > 1:
            self.resize = layers.ResizeTransform(self.int_downsize, 3)
            self.fullsize = layers.ResizeTransform(1 / self.int_downsize, 3)
        else:
            self.resize = None
            self.fullsize = None

        down_shape = [int(dim / self.int_downsize) for dim in self.img_size]
        self.integrate = layers.VecInt(down_shape, self.int_steps) if self.int_steps > 0 else None

        self.transformer = layers.SpatialTransformer(self.img_size)

    def forward(self, x, y, x_seg=None, y_seg=None, registration=False):

        source, target = x, y
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        pos_flow = self.flow(d3)

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