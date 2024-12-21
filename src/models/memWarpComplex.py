import clip
import math
import numpy as np
import pandas as pd
from collections import OrderedDict

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
            nn.BatchNorm3d(out_cs),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class memWarpComplex(nn.Module):

    def __init__(self,
            start_channel = '32',     # N_s in the paper
            lk_size = '5',            # kernel size of LK encoder
            img_size = '(128,128,16)',# input image size
            s_factor = '0.1',         # segmentation factor
            n_phi = 1024,             # MLP hidden units
            n_slots = 4,              # number of memory slots
            is_int = '1',             # whether to use integration
        ):

        super(memWarpComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.img_size = eval(img_size)
        self.s_factor = float(s_factor)
        self.n_phi = int(n_phi)
        self.n_slots = int(n_slots)
        self.is_int = int(is_int)

        print("start_channel: %d, lk_size: %d, img_size: %s, s_factor: %.1f, n_phi: %d, n_slots: %d, is_int: %d" % (self.start_channel, self.lk_size, self.img_size, self.s_factor, self.n_phi, self.n_slots, self.is_int))

        N_s = self.start_channel
        ks = self.lk_size

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
            nn.Conv3d(N_s*2, N_s * 2, 3, 1, 1),
        )
        self.disp_field_1 = nn.Sequential(
            encoder(N_s * 4, N_s * 2),
            nn.Conv3d(N_s * 2, N_s * 2, 3, 1, 1),
        )
        self.disp_field_0 = nn.Sequential(
            encoder(N_s * 4, N_s * 2),
            nn.Conv3d(N_s * 2, N_s * 2, 3, 1, 1),
        )
        self.mem_warp = scFilters(self.n_phi, self.n_slots, N_s*2, N_s*2, self.s_factor)

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

    def forward(self, x, y, x_seg=None, y_seg=None, registration=False, is_erf=False):

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
        field_2_feas = self.disp_field_2(torch.cat((d1_y+d1_x, d1_y-d1_x), dim=1))
        logit_2_feas = d1_y
        field_2, logits_2, mem_filters_2 = self.mem_warp(field_2_feas, logit_2_feas)
        field_2 = flow_multiplier(field_2, 1/2)
        preint_flow_2 = field_2
        if self.is_int:
            field_2 = self.integrate_3(field_2)
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
        field_1_feas = self.disp_field_1(torch.cat((d2_y+d2_x, d2_y-d2_x), dim=1))
        logit_1_feas = d2_y
        field_1, logits_1, mem_filters_1 = self.mem_warp(field_1_feas, logit_1_feas)
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
        field_0_feas = self.disp_field_0(torch.cat((d3_y+d3_x, d3_y-d3_x), dim=1))
        logit_0_feas = d3_y
        field_0, logits_0, mem_filters_0 = self.mem_warp(field_0_feas, logit_0_feas)
        field_0 = flow_multiplier(field_0, 1/2)
        preint_flow_0 = field_0
        field_0 = field_0 + up_field_1

        int_flows = [preint_flow_0, preint_flow_1, preint_flow_2]
        pos_flows = [field_0, field_1, field_2]
        seg_logits = [logits_0, logits_1, logits_2]
        mem_filters = [mem_filters_0, mem_filters_1, mem_filters_2]

        if not registration:
            return int_flows, pos_flows, seg_logits, mem_filters
        else:
            return pos_flows[0], seg_logits[0]

def flow_multiplier(flow, multiplier):

    flow_x, flow_y, flow_z = torch.chunk(flow, 3, dim=1)
    flow_x, flow_y = flow_x * multiplier, flow_y * multiplier
    flow = torch.cat((flow_x, flow_y, flow_z), dim=1)

    return flow

class memorySlots(nn.Module):

    def __init__(self,
            n_phi = 1024,           # MLP hidden units
            n_slots = 13,           # number of memory slots
            flow_cs = 32,           # number of features in each memory slot
            segt_cs = 32,           # number of features in each memory slot
        ):
        super(memorySlots, self).__init__()

        self.n_phi = n_phi
        self.n_slots = n_slots
        self.flow_cs = flow_cs
        self.segt_cs = segt_cs

        self.generate_filters = nn.Sequential(
            nn.Linear(self.n_slots, n_phi),
            nn.ReLU(inplace=True),
            nn.Linear(n_phi, n_phi*2),
            nn.ReLU(inplace=True),
            nn.Linear(n_phi*2, flow_cs*3),
        )
        self.mapping_feas = nn.Sequential(
            encoder(flow_cs, flow_cs, 3, 1, 1),
            nn.Conv3d(flow_cs, flow_cs*3, 1, 1, 0)
        )
        self.init_mask_encoding()

    def init_mask_encoding(self):

        self.one_hot_tensor = torch.eye(self.n_slots).to('cuda')
        self.one_hot_tensor.requires_grad_(False)
        self.one_hot_tensor = self.one_hot_tensor.float() 

        print("----->>>> Successfully initialized **one-hot tensor** for %d classes" % (len(self.one_hot_tensor)))

    def forward(self, y):
        '''
        y: (b,flow_cs,h,w,d)
        '''
        new_y = self.mapping_feas(y) # (b, flow_cs*3, h, w, d)
        memory_filters = self.generate_filters(self.one_hot_tensor).unsqueeze(0) # (1, n_slots, flow_cs*3)
        memory_filters = F.normalize(memory_filters, p=2, dim=2) # (1, n_slots, flow_cs*3)
        memory_filters = memory_filters.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (1, n_slots, flow_cs*3, 1, 1, 1)
        # print(new_y.shape, memory_filters.shape)
        logits = (new_y.unsqueeze(1) * memory_filters).sum(2) # (b, n_slots, h, w, d)

        return memory_filters, logits

class scFilters(nn.Module):

    def __init__(self, 
        n_phi,
        n_slots,
        flow_cs,
        segt_cs,
        s_factor=0.1,
    ):
        super(scFilters, self).__init__()

        self.s_factor = s_factor
        self.get_memory_filters = memorySlots(n_phi, n_slots, flow_cs, segt_cs)
        self.flow = nn.Conv3d(flow_cs, 3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def compute_confidence(self, tensor, min_val=1e-9):

        tensor_clamped = tensor.clamp(min=min_val)
        entropy = -(tensor_clamped * torch.log(tensor_clamped)).sum(dim=1)
        entropy = entropy / np.log(tensor_clamped.shape[1])

        return 1.-entropy

    def forward(self, flow_feas, segt_feas):
        '''
        flow_feas: (b,flow_cs,h,w,d)
        segt_feas: (b,segt_cs,h,w,d)
        '''
        memory_filters, logits = self.get_memory_filters(segt_feas)
        probs = F.softmax(logits/self.s_factor, dim=1).unsqueeze(2) # (b, n_slots, 1, h, w, d)
        confidence = self.compute_confidence(probs)
        flow_filters = (memory_filters * probs).sum(1) # (b, flow_cs*3, h, w, d)

        flow_x_filters, flow_y_filters, flow_z_filters = torch.chunk(flow_filters, 3, dim=1) # (b, flow_cs, h, w, d)

        flow_x = (flow_x_filters * flow_feas).sum(1, keepdim=True) # (b, 1, h, w, d)
        flow_y = (flow_y_filters * flow_feas).sum(1, keepdim=True) # (b, 1, h, w, d)
        flow_z = (flow_z_filters * flow_feas).sum(1, keepdim=True) # (b, 1, h, w, d)
        scp_flow = torch.cat([flow_x, flow_y, flow_z], 1)

        flow = confidence*scp_flow + (1-confidence)*self.flow(flow_feas)

        # flow = scp_flow

        return flow, logits, memory_filters.squeeze()