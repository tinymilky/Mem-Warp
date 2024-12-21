import torch
import argparse
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from utils import getters, setters
from utils.loss import BinaryDiceLoss, Grad3d, BinaryDiceLoss
from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, dice_eval, GaussianBlur2D


def blur_axial(x, blur):

    xs = torch.chunk(x, 16, dim=-1)
    outs = []
    for x in xs:
        outs.append(blur(x.squeeze(-1)).unsqueeze(-1))
    return torch.cat(outs, dim=-1)

def all_downsample2x(x, y, x_seg, y_seg, blur):

    x = blur_axial(x, blur)
    y = blur_axial(y, blur)

    x = F.interpolate(x, scale_factor=(0.5,0.5,1), mode='trilinear', align_corners=True)
    y = F.interpolate(y, scale_factor=(0.5,0.5,1), mode='trilinear', align_corners=True)

    x_seg = F.interpolate(x_seg, scale_factor=(0.5,0.5,1), mode='trilinear', align_corners=True)
    y_seg = F.interpolate(y_seg, scale_factor=(0.5,0.5,1), mode='trilinear', align_corners=True)

    return x, y, x_seg, y_seg

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt)
    model_saver = getters.getModelSaver(opt)

    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)
    blur1 = GaussianBlur2D(channels=1, sigma=1).cuda()

    criterion_sim_0 = nn.MSELoss()
    criterion_sim_1 = nn.MSELoss()
    criterion_sim_2 = nn.MSELoss()

    criterion_reg = Grad3d(penalty='l2')
    criterion_dsc = BinaryDiceLoss()
    best_dsc = 0
    best_epoch = 0
    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        loss_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_dsc_0_all = AverageMeter()
        loss_sim_0_all = AverageMeter()
        loss_seg_all = AverageMeter()

        for idx, data in enumerate(train_loader):
            model.train()
            adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])
            data = [Variable(t.cuda().float()) for t in data[:5]]
            x, x_seg = data[0], data[1]
            y, y_seg = data[2], data[3]

            x_seg_oh = F.one_hot(x_seg.long(), opt['n_classes']).squeeze(1).permute(0,4,1,2,3).float()
            y_seg_oh = F.one_hot(y_seg.long(), opt['n_classes']).squeeze(1).permute(0,4,1,2,3).float()

            x_0, x_seg_oh_0, y_0, y_seg_oh_0 = x, x_seg_oh, y, y_seg_oh
            x_1, y_1, x_seg_oh_1, y_seg_oh_1 = all_downsample2x(x_0, y_0, x_seg_oh_0, y_seg_oh_0, blur1)
            x_2, y_2, x_seg_oh_2, y_seg_oh_2 = all_downsample2x(x_1, y_1, x_seg_oh_1, y_seg_oh_1, blur1)

            int_flows, pos_flows, seg_logits, _ = model(x, y)
            int_flow_0, int_flow_1, int_flow_2 = int_flows
            pos_flow_0, pos_flow_1, pos_flow_2 = pos_flows
            seg_logits_0, seg_logits_1, seg_logits_2 = seg_logits

            # regularization loss
            reg_loss_0 = criterion_reg(int_flow_0,y) * opt['reg_w']
            reg_loss_1 = criterion_reg(int_flow_1,y) * opt['reg_w'] / 2
            reg_loss_2 = criterion_reg(int_flow_2,y) * opt['reg_w'] / 4
            reg_loss = reg_loss_0 + reg_loss_1 + reg_loss_2
            loss_reg_all.update(reg_loss.item(), y.numel())

            # seg dice loss
            if opt['seg_w'] > 0:
                seg_mem_0 = F.softmax(seg_logits_0/opt['s_factor'], dim=1)
                seg_loss_0 = criterion_dsc(seg_mem_0, y_seg_oh_0) * opt['seg_w']
                seg_mem_1 = F.softmax(seg_logits_1/opt['s_factor'], dim=1)
                seg_loss_1 = criterion_dsc(seg_mem_1, y_seg_oh_1) * opt['seg_w'] / 2
                seg_mem_2 = F.softmax(seg_logits_2/opt['s_factor'], dim=1)
                seg_loss_2 = criterion_dsc(seg_mem_2, y_seg_oh_2) * opt['seg_w'] / 4
                seg_loss = seg_loss_0 + seg_loss_1 + seg_loss_2
            else:
                seg_loss = reg_loss * 0
            loss_seg_all.update(seg_loss.item(), y.numel())

            if opt['dsc_w'] == 0:
                dsc_loss = reg_loss * 0
            else:
                dsc_loss_0 = criterion_dsc(model.transformer_1(x_seg_oh_0.float(),pos_flow_0.float()), y_seg_oh_0) * opt['dsc_w']
                dsc_loss_1 = criterion_dsc(model.transformer_2(x_seg_oh_1.float(),pos_flow_1.float()), y_seg_oh_1) * opt['dsc_w'] / 2
                dsc_loss_2 = criterion_dsc(model.transformer_3(x_seg_oh_2.float(),pos_flow_2.float()), y_seg_oh_2) * opt['dsc_w'] / 4
                dsc_loss = dsc_loss_0 + dsc_loss_1 + dsc_loss_2
            loss_dsc_0_all.update(dsc_loss.item(), y.numel())

            sim_loss_0 = criterion_sim_0(model.transformer_1(x_0.float(),pos_flow_0.float()), y_0) * opt['sim_w']
            sim_loss_1 = criterion_sim_1(model.transformer_2(x_1.float(),pos_flow_1.float()), y_1) * opt['sim_w'] / 2
            sim_loss_2 = criterion_sim_2(model.transformer_3(x_2.float(),pos_flow_2.float()), y_2) * opt['sim_w'] / 4
            loss_sim_0_all.update(sim_loss_0.item(), y.numel())
            sim_loss = sim_loss_0 + sim_loss_1 + sim_loss_2

            loss = reg_loss + dsc_loss + sim_loss + seg_loss
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter [{}/{}] loss {:.4f}, reg: {:.4f}, dsc: {:.4f}, sim: {:.4f}, seg: {:.4f}'.format(idx+1, len(train_loader), loss.item(), reg_loss.item(), dsc_loss.item(), sim_loss.item(), seg_loss.item()), end='\r')

        print('Epoch {} train loss {:.4f}, REG {:.4f}, DSC {:.4f}, SIM {:.4f}, SEG {:.4f}'.format(epoch+1, loss_all.avg, loss_reg_all.avg, loss_dsc_0_all.avg, loss_sim_0_all.avg, loss_seg_all.avg))

        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        seg_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [Variable(t.cuda())  for t in data[:4]]
                x, x_seg = data[0].float(), data[1].float()
                y, y_seg = data[2].float(), data[3].float()

                pos_flow, seg_logits = model(x,y,registration=True)
                if opt['seg_w'] > 0:
                    pred_seg = F.softmax(seg_logits[:,:opt['n_classes'],...]/opt['s_factor'], dim=1)
                    pred_seg = torch.argmax(pred_seg, dim=1)
                    dsc = dice_eval(pred_seg.long(), y_seg.long(), opt['n_classes'])
                    seg_dsc.update(dsc.item(), x.size(0))

                def_out = reg_model(x_seg.cuda().float(), pos_flow)
                dsc = dice_eval(def_out.long(), y_seg.long(), opt['n_classes'])
                eval_dsc.update(dsc.item(), x.size(0))
                dsc = dice_eval(x_seg.long(), y_seg.long(), opt['n_classes'])
                init_dsc.update(dsc.item(), x.size(0))

                pos_flow, seg_logits = model(y,x,registration=True)
                if opt['seg_w'] > 0:
                    pred_seg = F.softmax(seg_logits[:,:opt['n_classes'],...]/opt['s_factor'], dim=1)
                    pred_seg = torch.argmax(pred_seg, dim=1)
                    dsc = dice_eval(pred_seg.long(), x_seg.long(), opt['n_classes'])
                    seg_dsc.update(dsc.item(), x.size(0))

                def_out = reg_model(y_seg.cuda().float(), pos_flow)
                dsc = dice_eval(def_out.long(), x_seg.long(), opt['n_classes'])
                eval_dsc.update(dsc.item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch
        model_saver.saveModel(model, epoch, eval_dsc.avg)
        print('Epoch {} init dice {:.4f}, eval dice {:.4f}, seg dice {:.4f}, best reg dice {:.4f} at epoch {}'.format(epoch+1, init_dsc.avg, eval_dsc.avg, seg_dsc.avg, best_dsc, best_epoch+1))

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 5,          # number of checkpoints to keep
        'power': 0.9,                # decay power
        'num_workers': 4,            # number of workers for data loading
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'vxmDense')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'acdcreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--epochs", type = int, default = 401)
    parser.add_argument("--sim_w", type = float, default = 1)
    parser.add_argument("--reg_w", type = float, default = 0.01)
    parser.add_argument("--dsc_w", type = float, default = 1)
    parser.add_argument("--seg_w", type = float, default = 1.)
    parser.add_argument("--lr", type = float, default = 4e-4)
    parser.add_argument("--img_size", type = str, default = '(128,128,16)')
    parser.add_argument("--n_classes", type = int, default = 4) # 4 for cardiac, 14 for abdomen, and background labeled as 0
    parser.add_argument("--s_factor", type = float, default = 0.1)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    print("sim: {}, reg: {}, dsc: {}, seg: {}".format(opt['sim_w'], opt['reg_w'], opt['dsc_w'], opt['seg_w']))

    run(opt)

'''
python train_cardiac_memwarp.py -m memWarpComplexDw1Sw1S32Lk5 -d acdcreg -bs 4 --dsc_w 1. --seg_w 1. start_channel=32 lk_size=5

python train_cardiac_memwarp.py -m lapWarpComplexS32Dw0Sw0S32Lk5 -d acdcreg -bs 4 --dsc_w 0. --seg_w 0 start_channel=32 lk_size=5
'''