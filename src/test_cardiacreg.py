import os
import torch
import interpol
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from torch.autograd import Variable

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, dice_binary, dice_eval, computeJacDetVal, computeSDLogJ, compute_HD95, jacobian_determinant 

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    test_loader = getters.getDataLoader(opt, split='test')
    model, _ = getters.getTestModelWithCheckpoints(opt)
    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()

    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_lv_dsc = AverageMeter()
    init_lv_dsc = AverageMeter()
    eval_rv_dsc = AverageMeter()
    init_rv_dsc = AverageMeter()
    eval_lvm_dsc = AverageMeter()
    init_lvm_dsc = AverageMeter()
    eval_jac_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()

    df_data = []
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            sub_idx = data[4].item()
            data = [Variable(t.cuda())  for t in data[:4]]
            x, x_seg = data[0].float(), data[1]
            y, y_seg = data[2].float(), data[3]

            warped_x2y, pos_flow1 = model(x,y,registration=True)
            warped_y2x, pos_flow2 = model(y,x,registration=True)

            df_row = []

            def_out1 = reg_model(x_seg.cuda().float(), pos_flow1)
            dsc1, rv_dsc1, lvm_dsc1, lv_dsc1 = dice_eval(def_out1.long(), y_seg.long(), 4, output_individual=True)
            def_out2 = reg_model(y_seg.cuda().float(), pos_flow2)
            dsc2, rv_dsc2, lvm_dsc2, lv_dsc2 = dice_eval(def_out2.long(), x_seg.long(), 4, output_individual=True)
            dsc_p = (dsc1 + dsc2) / 2
            rv_dsc = (rv_dsc1 + rv_dsc2) / 2
            lvm_dsc = (lvm_dsc1 + lvm_dsc2) / 2
            lv_dsc = (lv_dsc1 + lv_dsc2) / 2
            eval_dsc.update(dsc_p.item(), x.size(0))
            eval_lv_dsc.update(lv_dsc.item(), x.size(0))
            eval_rv_dsc.update(rv_dsc.item(), x.size(0))
            eval_lvm_dsc.update(lvm_dsc.item(), x.size(0))

            dsc_i, rv_dsci, lvm_dsci, lv_dsci = dice_eval(x_seg.long(), y_seg.long(), 4, output_individual=True)
            init_dsc.update(dsc_i.item(), x.size(0))
            init_lv_dsc.update(lv_dsci.item(), x.size(0))
            init_rv_dsc.update(rv_dsci.item(), x.size(0))
            init_lvm_dsc.update(lvm_dsci.item(), x.size(0))

            df_row.append(sub_idx)
            df_row.append(dsc_p.item())
            df_row.append(dsc_i.item())
            df_row.append(rv_dsc.item())
            df_row.append(lvm_dsc.item())
            df_row.append(lv_dsc.item())

            # Jacobian determinant
            jac_det1 = jacobian_determinant(pos_flow1.detach().cpu().numpy())
            jac_det2 = jacobian_determinant(pos_flow2.detach().cpu().numpy())
            jac_det_val1 = computeJacDetVal(jac_det1, x_seg.shape[2:])
            jac_det_val2 = computeJacDetVal(jac_det2, x_seg.shape[2:])
            jac_det_val = (jac_det_val1 + jac_det_val2) / 2
            eval_jac_det.update(jac_det_val, x.size(0))

            # Standard deviation of log Jacobian determinant
            std_dev_jac1 = computeSDLogJ(jac_det1)
            std_dev_jac2 = computeSDLogJ(jac_det2)
            std_dev_jac = (std_dev_jac1 + std_dev_jac2) / 2
            eval_std_det.update(std_dev_jac, x.size(0))

            # Hausdorff distance 95
            moving = x_seg.long().squeeze().cpu().numpy()
            fixed = y_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out1.long().squeeze().cpu().numpy()
            hd95_1 = compute_HD95(moving, fixed, moving_warped, 4, opt['voxel_spacing'])
            init_hd95_1 = compute_HD95(moving, fixed, moving, 4, opt['voxel_spacing'])

            moving = y_seg.long().squeeze().cpu().numpy()
            fixed = x_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out2.long().squeeze().cpu().numpy()
            hd95_2 = compute_HD95(moving, fixed, moving_warped, 4, opt['voxel_spacing'])
            init_hd95_2 = compute_HD95(moving, fixed, moving, 4, opt['voxel_spacing'])

            hd95 = (hd95_1 + hd95_2) / 2
            eval_hd95.update(hd95, x.size(0))
            init_hd95_ = (init_hd95_1 + init_hd95_2) / 2
            init_hd95.update(init_hd95_, x.size(0))

            df_row.append(jac_det_val)
            df_row.append(std_dev_jac)
            df_row.append(hd95)
            df_row.append(init_hd95_)
            df_data.append(df_row)

            print("Subject {} dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f}, jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}".format(sub_idx, dsc_p, dsc_i, rv_dsc, lvm_dsc, lv_dsc, jac_det_val, std_dev_jac, hd95, init_hd95_))

            if opt['is_save']:
                pos_flow1 = pos_flow1.permute(2,3,4,0,1).cpu().numpy()
                pos_flow2 = pos_flow2.permute(2,3,4,0,1).cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'flow_fields')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(pos_flow1, None, None), os.path.join(fp,'%s_flow_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(pos_flow2, None, None), os.path.join(fp,'%s_flow_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                warped_x2y = warped_x2y.squeeze().cpu().numpy()
                warped_y2x = warped_y2x.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'warped_images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(warped_x2y, None, None), os.path.join(fp,'%s_warped_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(warped_y2x, None, None), os.path.join(fp,'%s_warped_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                x = x.squeeze().cpu().numpy()
                y = y.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(x, None, None), os.path.join(fp,'%s_x.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(y, None, None), os.path.join(fp,'%s_y.nii.gz' % (str(sub_idx).zfill(3))))

    print("init dice: {:.7f}, init rv dice: {:.7f}, init lvm dice: {:.7f}, init lv dice: {:.7f}".format(init_dsc.avg, init_rv_dsc.avg, init_lvm_dsc.avg, init_lv_dsc.avg))

    print("Average dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f}, jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}".format(eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg, eval_lv_dsc.avg, eval_jac_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg))

    df_row = ['Average', eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg, eval_lv_dsc.avg, eval_jac_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg]
    df_data.append(df_row)

    keys = ['subject', 'dice', 'init_dice', 'rv_dice', 'lvm_dice', 'lv_dice', 'jac_det', 'std_dev_jac', 'hd95', 'init_hd95']
    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_%s.csv' % opt['model'])
    df.to_csv(fp, index=False)

if __name__ == '__main__':

    opt = {
        'img_size': (128,128,16),  # input image size
        'logs_path': './logs',     # path to saved logs
        'num_workers': 4,          # number of workers for data loading
        'voxel_spacing': (1.8,1.8,10),# voxel size
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'voxelMorphComplex')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'acdcreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--load_ckpt", type = str, default = "best") # best, last or epoch
    parser.add_argument("--is_save", type = int, default = 0) # whether to save the flow field

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}

    run(opt)

'''
python test_cardiacreg_memwarp.py -m memWarpComplexDw1Sw1S32Lk5 -d acdcreg --datasets_path ./../../../data/ start_channel=32 lk_size=5
'''