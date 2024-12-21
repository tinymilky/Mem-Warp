import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from models.backbones.voxelmorph.torch.layers import SpatialTransformer, VecInt, ResizeTransform

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
        ks = [kernel_size, kernel_size, 3]
        pd = [padding, padding, 1]
        self.large = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, ks, stride, pd),
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
            nn.BatchNorm3d(out_cs),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class lapWarpComplex(nn.Module):

    def __init__(self,
            start_channel = '32',  # N_s in the paper
            lk_size = '5',         # kernel size of LK encoder
            img_size = '(128,128,16)', # input image size
            is_int = '1',        # whether to integrate the flow field
        ):

        super(lapWarpComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.img_size = eval(img_size)
        self.is_int = int(is_int)

        print("start_channel: %d, lk_size: %d, img_size: %s, is_int: %d" % (self.start_channel, self.lk_size, self.img_size, self.is_int))

        N_s = self.start_channel
        ks = self.lk_size

        self.flow = nn.Conv3d(self.start_channel*2,3,3,1,1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.eninput = encoder(1, N_s)
        self.ec1 = encoder(N_s, N_s)
        self.ec2 = encoder(N_s, N_s * 2, 3, (2,2,1), 1) # stride=2
        self.ec3 = encoder(N_s * 2, N_s * 2, ks, 1, ks//2) # LK encoder
        self.ec4 = encoder(N_s * 2, N_s * 4, 3, (2,2,1), 1) # stride=2
        self.ec5 = encoder(N_s * 4, N_s * 4, ks, 1, ks//2) # LK encoder
        self.ec6 = encoder(N_s * 4, N_s * 8, 3, (2,2,1), 1) # stride=2
        self.ec7 = encoder(N_s * 8, N_s * 8, ks, 1, ks//2) # LK encoder
        self.ec8 = encoder(N_s * 8, N_s * 8, 3, (2,2,1), 1) # stride=2
        self.ec9 = encoder(N_s * 8, N_s * 8, ks, 1, ks//2) # LK encoder

        self.dc1 = encoder(N_s * 16, N_s * 8, 3, 1, 1)
        self.dc2 = encoder(N_s * 8,  N_s * 4, ks, 1, ks//2)
        self.dc3 = encoder(N_s * 8,  N_s * 4, 3, 1, 1)
        self.dc4 = encoder(N_s * 4,  N_s * 2, ks, 1, ks//2)
        self.dc5 = encoder(N_s * 4,  N_s * 4, 3, 1, 1)
        self.dc6 = encoder(N_s * 4,  N_s * 2, ks, 1, ks//2)
        self.dc7 = encoder(N_s * 3,  N_s * 2, 3, 1, 1)
        self.dc8 = encoder(N_s * 2,  N_s * 2, ks, 1, ks//2)

        self.up1 = decoder(N_s * 8, N_s * 8, kernel_size=(2,2,1),stride=(2,2,1))
        self.up2 = decoder(N_s * 4, N_s * 4, kernel_size=(2,2,1),stride=(2,2,1))
        self.up3 = decoder(N_s * 2, N_s * 2, kernel_size=(2,2,1),stride=(2,2,1))
        self.up4 = decoder(N_s * 2, N_s * 2, kernel_size=(2,2,1),stride=(2,2,1))

        self.disp_field_4 = nn.Sequential(
            encoder(N_s * 16, N_s * 8),
            nn.Conv3d(N_s*8, 3, 3, 1, 1),
        )
        self.disp_field_3 = nn.Sequential(
            encoder(N_s * 8, N_s * 4),
            nn.Conv3d(N_s*4, 3, 3, 1, 1),
        )
        self.disp_field_2 = nn.Sequential(
            encoder(N_s * 4, N_s * 2),
            nn.Conv3d(N_s*2, 3, 3, 1, 1),
        )
        self.disp_field_1 = nn.Sequential(
            encoder(N_s * 4, N_s * 2),
            nn.Conv3d(N_s * 2, 3, 3, 1, 1),
        )
        self.disp_field_0 = nn.Sequential(
            encoder(N_s * 4, N_s * 2),
            nn.Conv3d(N_s * 2, 3, 3, 1, 1),
        )
        self.initialize_layer(self.disp_field_4)
        self.initialize_layer(self.disp_field_3)
        self.initialize_layer(self.disp_field_2)
        self.initialize_layer(self.disp_field_1)
        self.initialize_layer(self.disp_field_0)

        ss = self.img_size
        self.transformer_5 = SpatialTransformer([s//16 for s in ss[:2]]+[ss[2]])
        self.transformer_4 = SpatialTransformer([s//8 for s in ss[:2]]+[ss[2]])
        self.transformer_3 = SpatialTransformer([s//4 for s in ss[:2]]+[ss[2]])
        self.transformer_2 = SpatialTransformer([s//2 for s in ss[:2]]+[ss[2]])
        self.transformer_1 = SpatialTransformer([s//1 for s in ss[:2]]+[ss[2]])

        if self.is_int:
            self.integrate_5 = VecInt([s//16 for s in ss[:2]]+[ss[2]], 7)
            self.integrate_4 = VecInt([s//8 for s in ss[:2]]+[ss[2]], 7)
            self.integrate_3 = VecInt([s//4 for s in ss[:2]]+[ss[2]], 7)
            self.integrate_2 = VecInt([s//2 for s in ss[:2]]+[ss[2]], 7)
            self.integrate_1 = VecInt([s//1 for s in ss[:2]]+[ss[2]], 7)

        self.up_tri = torch.nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)

    def initialize_layer(self, module):
        for child in module.children():
            if isinstance(child, nn.Conv3d):
                torch.nn.init.constant_(child.weight, 0)
                if child.bias is not None:
                    torch.nn.init.constant_(child.bias, 0)

    def forward(self, x, y, registration=False):

        x_in = torch.cat((x, y), dim=0)

        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)
        e0_x, e0_y = torch.chunk(e0, 2, dim=0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)
        e1_x, e1_y = torch.chunk(e1, 2, dim=0)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)
        e2_x, e2_y = torch.chunk(e2, 2, dim=0)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)
        e3_x, e3_y = torch.chunk(e3, 2, dim=0)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)
        e4_x, e4_y = torch.chunk(e4, 2, dim=0)
        field_4 = self.disp_field_4(torch.cat((e4_y+e4_x, e4_y-e4_x), dim=1))
        field_4 = flow_multiplier(field_4, 1/2)
        preint_flow_4 = field_4
        if self.is_int:
            field_4 = self.integrate_5(field_4)
        warped_e4_x = self.transformer_5(e4_x, field_4)
        e4 = torch.cat((warped_e4_x, e4_y), dim=0)
        up_field_4 = self.up_tri(field_4)
        up_field_4 = flow_multiplier(up_field_4, 2)
        warped_e3_x = self.transformer_4(e3_x, up_field_4)
        e3 = torch.cat((warped_e3_x, e3_y), dim=0)

        d0 = torch.cat((self.up1(e4), e3), dim=1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)
        d0_x, d0_y = torch.chunk(d0, 2, dim=0)
        field_3 = self.disp_field_3(torch.cat((d0_y+d0_x, d0_y-d0_x), dim=1))
        field_3 = flow_multiplier(field_3, 1/2)
        preint_flow_3 = field_3
        if self.is_int:
            field_3 = self.integrate_4(field_3)
        warped_d0_x = self.transformer_4(d0_x, field_3)
        d0 = torch.cat((warped_d0_x, d0_y), dim=0)
        field_3 = field_3 + up_field_4
        up_field_3 = self.up_tri(field_3)
        up_field_3 = flow_multiplier(up_field_3, 2)
        warped_e2_x = self.transformer_3(e2_x, up_field_3)
        e2 = torch.cat((warped_e2_x, e2_y), dim=0)

        d1 = torch.cat((self.up2(d0), e2), dim=1)
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)
        d1_x, d1_y = torch.chunk(d1, 2, dim=0)
        field_2 = self.disp_field_2(torch.cat((d1_y+d1_x, d1_y-d1_x), dim=1))
        preint_flow_2 = field_2
        if self.is_int:
            field_2 = self.integrate_3(field_2)
        field_2 = flow_multiplier(field_2, 1/2)
        warped_d1_x = self.transformer_3(d1_x, field_2)
        d1 = torch.cat((warped_d1_x, d1_y), dim=0)
        field_2 = field_2 + up_field_3
        up_field_2 = self.up_tri(field_2)
        up_field_2 = flow_multiplier(up_field_2, 2)
        warped_e1_x = self.transformer_2(e1_x, up_field_2)
        e1 = torch.cat((warped_e1_x, e1_y), dim=0)

        d2 = torch.cat((self.up3(d1), e1), dim=1)
        d2 = self.dc5(d2)
        d2 = self.dc6(d2)
        d2_x, d2_y = torch.chunk(d2, 2, dim=0)
        field_1 = self.disp_field_1(torch.cat((d2_y+d2_x, d2_y-d2_x), dim=1))
        field_1 = flow_multiplier(field_1, 1/2)
        preint_flow_1 = field_1
        if self.is_int:
            field_1 = self.integrate_2(field_1)
        warped_d2_x = self.transformer_2(d2_x, field_1)
        d2 = torch.cat((warped_d2_x, d2_y), dim=0)
        field_1 = field_1 + up_field_2
        up_field_1 = self.up_tri(field_1)
        up_field_1 = flow_multiplier(up_field_1, 2)
        warped_e0_x = self.transformer_1(e0_x, up_field_1)
        e0 = torch.cat((warped_e0_x, e0_y), dim=0)

        d3 = torch.cat((self.up4(d2), e0), dim=1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)
        d3_x, d3_y = torch.chunk(d3, 2, dim=0)
        field_0 = self.disp_field_0(torch.cat((d3_y+d3_x, d3_y-d3_x), dim=1))
        field_0 = flow_multiplier(field_0, 1/2)
        preint_flow_0 = field_0
        field_0 = field_0 + up_field_1

        int_flows = preint_flow_0, preint_flow_1, preint_flow_2
        pos_flows = field_0, field_1, field_2

        if not registration:
            return int_flows, pos_flows, int_flows, pos_flows
        else:
            return pos_flows[0], pos_flows[0]

def flow_multiplier(flow, multiplier):

    flow_x, flow_y, flow_z = torch.chunk(flow, 3, dim=1)
    flow_x, flow_y = flow_x * multiplier, flow_y * multiplier
    flow = torch.cat((flow_x, flow_y, flow_z), dim=1)

    return flow