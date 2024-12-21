import torch
import argparse
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from utils.loss import DiceLoss, Grad3d
from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, dice_eval

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

    criterion_sim = nn.MSELoss()
    criterion_reg = Grad3d(penalty='l2')
    criterion_dsc = DiceLoss(num_class=opt['n_classes'])
    best_dsc = 0
    best_epoch = 0
    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        loss_all = AverageMeter()
        loss_sim_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_dsc_all = AverageMeter()

        for idx, data in enumerate(train_loader):
            model.train()
            adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])

            data = [Variable(t.cuda()) for t in data[:4]]
            x, y = data[0].float(), data[2].float()
            x_seg, y_seg = data[1].float(), data[3].float()

            warped_x, preint_flow, pos_flow = model(x, y)

            # similarity loss
            sim_loss = criterion_sim(warped_x, y) * opt['sim_w']
            loss_sim_all.update(sim_loss.item(), y.numel())

            # regularisation loss
            reg_loss = criterion_reg(preint_flow,y)
            reg_loss = reg_loss * opt['reg_w']
            loss_reg_all.update(reg_loss.item(), y.numel())

            # dice loss
            if opt['dsc_w'] > 0:
                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=opt['n_classes'])
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
                def_seg = model.transformer(x_seg_oh.float(), pos_flow.float())
                dsc_loss = criterion_dsc(def_seg, y_seg) * opt['dsc_w']
            else:
                dsc_loss = sim_loss*0
            loss_dsc_all.update(dsc_loss.item(), y.numel())

            loss = sim_loss + reg_loss + dsc_loss

            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Sim: {:.5f}, DSC: {:.5f}'.format(idx+1, len(train_loader), loss.item(), sim_loss.item(), dsc_loss.item()), end='\r', flush=True)

        print('---->>>> Epoch {} train loss {:.4f}, Sim {:.5f}, DSC {:.5f}'.format(epoch+1, loss_all.avg, loss_sim_all.avg, loss_dsc_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [Variable(t.cuda())  for t in data[:4]]
                x, x_seg = data[0].float(), data[1].float()
                y, y_seg = data[2].float(), data[3].float()

                _, pos_flow = model(x,y,registration=True)

                # def_out = model.align_img(grid.float(), x_seg.float(), mode='nearest')
                def_out = reg_model(x_seg, pos_flow)
                dsc = dice_eval(def_out.long(), y_seg.long(), opt['n_classes'])
                eval_dsc.update(dsc.item(), x.size(0))
                dsc = dice_eval(x_seg.long(), y_seg.long(), opt['n_classes'])
                init_dsc.update(dsc.item(), x.size(0))

                _, pos_flow = model(y,x,registration=True)

                def_out = reg_model(y_seg, pos_flow)
                dsc = dice_eval(def_out.long(), x_seg.long(), opt['n_classes'])
                eval_dsc.update(dsc.item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch
        print('Epoch {} val dice {:.4f}, init dice {:.4f}, best dice {:.4f} at epoch {}'.format(epoch+1, eval_dsc.avg, init_dsc.avg, best_dsc, best_epoch+1))

        model_saver.saveModel(model, epoch, eval_dsc.avg)

        loss_all.reset()

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',       # path to save logs
        'save_freq': 2,              # save model every save_freq epochs
        'n_checkpoints': 6,          # number of checkpoints to keep
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
    parser.add_argument("--sim_w", type = float, default = 1.)
    parser.add_argument("--reg_w", type = float, default = 0.01)
    parser.add_argument("--dsc_w", type = float, default = 1.)
    parser.add_argument("--lr", type = float, default = 4e-4)
    parser.add_argument("--img_size", type = str, default = '(128,128,16)')
    parser.add_argument("--n_classes", type = int, default = 4)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    print("sim: {}, dsc: {}, reg: {}".format(opt['sim_w'], opt['dsc_w'], opt['reg_w']))

    run(opt)

'''
python train_cardiacreg.py -m lkunetComplexDw1Lk5 -d acdcreg -bs 4 --dsc_w 1. lk_size=5 
python train_cardiacreg.py -m voxelMorphComplexDw1 -d acdcreg -bs 4 lk_size=5 --dsc_w 1.
'''