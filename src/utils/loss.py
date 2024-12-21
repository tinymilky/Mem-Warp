import math
import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from models.backbones.voxelmorph.torch.layers import SpatialTransformer

class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3] or y_pred.shape[4] != y_true.shape[4]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3], y_true.shape[4]), mode='trilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        else:
            union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class DiceLoss2D(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 3, 1, 2).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3]), mode='bilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3])

        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3]) + torch.pow(y_true, 2).sum(dim=[2, 3])
        else:
            union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class BinaryDiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=(2,3,4))
        union = y_pred.sum(dim=(2,3,4)) + y_true.sum(dim=(2,3,4))
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class Grad3d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad3d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad

class Grad2d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad2d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad


class NccLoss(nn.Module):

    def __init__(self, win=None):
        super(NccLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1.-torch.mean(cc)
