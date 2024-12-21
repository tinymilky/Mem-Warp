import os
import torch
import random
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset


class acdcreg_loader(Dataset): # acdcreg_loader

    train_list = [idx for idx in range(1, 17+1)] + [idx for idx in range(21, 37+1)] \
               + [idx for idx in range(41, 57+1)] + [idx for idx in range(61, 77+1)] \
               + [idx for idx in range(81, 97+1)]

    val_list = [idx for idx in range(18, 21)] + [idx for idx in range(38, 41)] \
             + [idx for idx in range(58, 61)] + [idx for idx in range(78, 81)] \
             + [idx for idx in range(98, 100+1)]

    test_list = [idx for idx in range(101, 150+1)]

    def __init__(self,
            root_dir = './../../../data/acdcreg/',
            split = 'train', # train, val or test
            intensity_cap = 0.001,
            enable_random_ed_es_flip = 1,
        ):
        self.root_dir = root_dir
        self.split = split
        self.intensity_cap = intensity_cap
        self.enable_random_ed_es_flip = enable_random_ed_es_flip

        if self.split == 'train':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.train_list
        elif self.split == 'val':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.val_list
        elif self.split == 'test':
            self.root_dir = os.path.join(self.root_dir, 'test/')
            self.idxs = self.test_list
        else:
            raise ValueError('Invalid split name')

        self.init_augmentation()
        self.init_dataset_in_memory()

    def init_augmentation(self):

        self.random_flip = tio.RandomFlip(axes=(0,1))
        self.random_gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3))

    def init_dataset_in_memory(self):

        self.data = []
        for sub_idx in self.idxs:
            sub_idx_str = 'patient'+str(sub_idx).zfill(3)
            ed_img_fp = os.path.join(self.root_dir, sub_idx_str+'_ed_img.nii.gz')
            es_img_fp = os.path.join(self.root_dir, sub_idx_str+'_es_img.nii.gz')
            ed_seg_fp = os.path.join(self.root_dir, sub_idx_str+'_ed_seg.nii.gz')
            es_seg_fp = os.path.join(self.root_dir, sub_idx_str+'_es_seg.nii.gz')

            ed_img = nib.load(ed_img_fp).get_data()
            es_img = nib.load(es_img_fp).get_data()
            ed_seg = nib.load(ed_seg_fp).get_data()
            es_seg = nib.load(es_seg_fp).get_data()

            sub = {
                'sub_idx': sub_idx,
                'ed_img': ed_img,
                'es_img': es_img,
                'ed_seg': ed_seg,
                'es_seg': es_seg,
            }
            self.data.append(sub)
            print('sub_idx %s stored in memory' % sub_idx)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        sub = self.data[idx]
        ed_img, es_img, ed_seg, es_seg = sub['ed_img'], sub['es_img'], sub['ed_seg'], sub['es_seg']
        x, x_seg = ed_img, ed_seg
        y, y_seg = es_img, es_seg

        if self.enable_random_ed_es_flip and random.random() > 0.5:
            # randomly exchange ed and es
            x, y = es_img, ed_img
            x_seg, y_seg = es_seg, ed_seg

        x, x_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg)
        y, y_seg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg)

        # normalize to [0, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        x, x_seg = x[None,...], x_seg[None,...]
        y, y_seg = y[None,...], y_seg[None,...]

        x, x_seg = torch.from_numpy(x), torch.from_numpy(x_seg)
        y, y_seg = torch.from_numpy(y), torch.from_numpy(y_seg)

        # # Augmentation
        if self.split == 'train' and self.enable_random_ed_es_flip:
            # Flip augmentation
            subject = tio.Subject(
                img1 = tio.ScalarImage(tensor=x),
                img2 = tio.ScalarImage(tensor=y),
                msk1 = tio.LabelMap(tensor=x_seg),
                msk2 = tio.LabelMap(tensor=y_seg),
            )
            subject = self.random_flip(subject)
            x, x_seg = subject.img1.data, subject.msk1.data
            y, y_seg = subject.img2.data, subject.msk2.data

            # # Gamma augmentation
            # subject1 = tio.Subject(img = tio.ScalarImage(tensor=x))
            # subject2 = tio.Subject(img = tio.ScalarImage(tensor=y))
            # subject1 = self.random_gamma(subject1)
            # subject2 = self.random_gamma(subject2)
            # x, y = subject1.img.data, subject2.img.data

        return x, x_seg, y, y_seg, sub['sub_idx']