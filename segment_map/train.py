#!/usr/bin/python
# -*- encoding: utf-8 -*-




import os

from utils.loss import OhemCELoss
from findBest import findbest
from utils.optimizer import Optimizer
from model.swiftnet.models.semseg import SemsegModel
from model.swiftnet.models.resnet.resnet_single_scale import resnet18
from utils.logger import setup_logger
from loader.LGE import LGE

import torch
from torch.utils.data import DataLoader

import os.path as osp
import logging
import time
import datetime
import argparse



def train(args, expnum):

    logger = logging.getLogger()

    checkpoint_base = args.checkpoint_path
    datapth = args.data_path
    n_workers = 8

    checkpoint_path = os.path.join(checkpoint_base, args.explabel, expnum)

    if not osp.exists(checkpoint_path): os.makedirs(checkpoint_path)
    setup_logger(checkpoint_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ## dataset
    n_classes = 7
    n_img_per_gpu = 8

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    power = 0.9
    warmup_steps = 500
    warmup_start_lr = 1e-5


    cropsize = [540, 960]

    ## data loader
       
    ds = LGE(datapth, cropsize=cropsize, mode='train')
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)


    ## model
    ## load backbone and semantic segmentation model

    resnet = resnet18(pretrained=True, efficient=False)
    net = SemsegModel(resnet, n_classes)

    ## for using pre-trained weights that trained on another dataset
    if args.pre_trained == True:
        print('load cityscapes pretrained weights')
        state_dict = torch.load('./model_best.pt')
        r_weight = state_dict['logits.conv.weight'][0:7, :, :, :]
        state_dict['logits.conv.weight'] = r_weight
        r_weight = state_dict['logits.conv.bias'][0:7]
        state_dict['logits.conv.bias'] = r_weight
        net.load_state_dict(state_dict, strict=False)

   
    


    logger.info(net)
    net.to(device)
    net.train()

    ## loss function
    score_thres = 0.7
    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//16
  
    ignore_idx = 255
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)


    ## optimizer
    optim = Optimizer(
            model = net,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    it = 0
    flag=True
    while it < max_iter and flag:
        for (im, lb) in dl:
            im = im.to(device)
            lb = lb.to(device)
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()

            out,_ = net(im,(960, 540),(960,540))
                    
            loss = criteria_p(out, lb)
           
            loss.backward()
            optim.step()

            loss_avg.append(loss.item())
            ## print training log message
            if (it+1)%msg_iter==0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                lr = optim.lr
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                eta = int((max_iter - it) * (glob_t_intv / it))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join([
                        'it: {it}/{max_it}',
                        'lr: {lr:4f}',
                        'loss: {loss:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        it = it+1,
                        max_it = max_iter,
                        lr = lr,
                        loss = loss_avg,
                        time = t_intv,
                        eta = eta
                    )
                logger.info(msg)
                loss_avg = []
                st = ed
                ## save checkpoints
                for j in range(1, 20):
                    if it+1 == max_iter-msg_iter*j:
                        save_pth = osp.join(checkpoint_path, str(it+1) + '_model.pth')
                        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                        torch.save(state, save_pth)
                        logger.info('model saved to: {}'.format(save_pth))

            it = it + 1
            if it == max_iter:
                flag = False
                break

    ## dump the final model
    save_pth = osp.join(checkpoint_path, 'final_model.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(net)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":


    expnum = str(datetime.datetime.now().strftime('%m%d-%H%M%S'))
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--Sim',
        dest='Sim',
        type=int,
        default=0,
    )
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=20000,
    )
    parse.add_argument(
        '--explabel',
        dest='explabel',
        type=str,
        default='LGE_swift',
    )
    parse.add_argument(
        '--pre_trained',
        action='store_true',
        default=False
    )
    parse.add_argument(
        '--checkpoint_path',
        dest='checkpoint_path',
        type=str,
        default='./results/'
    )
    parse.add_argument(
        '--data_path',
        dest='data_path',
        type=str,
        default='/mnt/hdd1/data/LGE_dataset/'
    )

    args = parse.parse_args()

    train(args, expnum)

    findbest(args, expnum)