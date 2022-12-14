#!/usr/bin/python
# -*- encoding: utf-8 -*-

from utils.logger import setup_logger



from model.swiftnet.models.semseg import SemsegModel
from model.swiftnet.models.resnet.resnet_single_scale import resnet18

from loader.LGE import LGE

import torch
from torch.utils.data import DataLoader

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import argparse
import timeit



class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            n_classes = 7,
            lb_ignore = 255,
            cropsize = 512,
            *args, **kwargs):
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.cropsize = cropsize
        self.dl = dataloader
        self.net = model

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self):
        ## evaluate
        time_todevice = 0
        time_forward = 0
        time_max = 0
        time_tocpu = 0
        total_img = 0

        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        accuracy = 0

        with torch.no_grad():
            for i, (imgs, label) in enumerate(dloader):
                # N, _, H, W = label.shape
                # print(imgs.size())
                # A = timeit.default_timer()
                A = time.perf_counter()
                imgs = imgs.cuda()
                torch.cuda.synchronize()
                # B = timeit.default_timer()
                B = time.perf_counter()
                prob,_= self.net(imgs,(960, 540),(960,540))
                torch.cuda.synchronize()
                # C = timeit.default_timer()
                C = time.perf_counter()

                ##
                if 1:
                    prod_device = prob.max(1)[1]
                    torch.cuda.synchronize()
                    # D = timeit.default_timer()
                    D = time.perf_counter()
                    preds = prod_device.data.cpu().numpy()
                    torch.cuda.synchronize()
                    # E = timeit.default_timer()
                    E = time.perf_counter()
                    # preds = preds.numpy()

                    if i>10:
                        time_todevice = time_todevice + B-A
                        time_forward = time_forward + C-B
                        time_max = time_max + D - C
                        time_tocpu = time_tocpu + E-D

                else:
                    prob = prob.data.cpu().numpy()
                    D = timeit.default_timer()
                    preds = np.argmax(prob, axis=1)
                    E = timeit.default_timer()

                    if i > 10:
                        time_todevice = time_todevice + B - A
                        time_forward = time_forward + C - B
                        time_tocpu = time_tocpu + D - C
                        time_max = time_max + E - D


                # print(C-B)
                hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
                total = label.size(0)*label.size(1)*label.size(2)
                correct = (preds==label.data.numpy()).sum().item()

                accuracy += (correct/total)
                IOU = np.nanmean(np.diag(hist_once) / (np.sum(hist_once, axis=0)+np.sum(hist_once, axis=1)-np.diag(hist_once)))

                hist = hist + hist_once
                if i > 10:
                    total_img = total_img + 1


        accuracy = accuracy/total_img
              
        time_todevice = time_todevice/total_img
        time_forward = time_forward/total_img
        time_tocpu = time_tocpu/total_img
        time_max = time_max/total_img

        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        print(accuracy)
        print(IOUs)
        mIOU = np.mean(IOUs)
        return mIOU, time_todevice, time_forward, time_max, time_tocpu


def findbest(args, expnum):
    ## setting

    respth_base = args.checkpoint_path
    datapth =  args.data_path
    n_workers = 8
    

    respth = os.path.join(respth_base, args.explabel, expnum)



    # logger
    logger = logging.getLogger()
    setup_logger(respth)

    ## model
    print('\n')
    print('===='*20)
    print('evaluating the model ...\n')
    print('setup and restore model')
    n_classes = 7

    resnet = resnet18(pretrained=True, efficient=False)
    net = SemsegModel(resnet, n_classes)

    ## dataset
    batchsize = 1
    n_workers = 16
    dsval = LGE(datapth, mode='val')
    dl = DataLoader(dsval,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=n_workers,
                    drop_last=False)

    best_mIoU = 0
    best_iter = 0

    iter_start = args.max_iter - 950
    iter_end = args.max_iter

    
    save_pth = osp.join(respth, str(iter_start)+ '_model.pth')
    logger.info('final')
    net.load_state_dict(torch.load(save_pth,map_location='cuda:0'))
    net.cuda()
    net.eval()

    ## evaluator
    print('compute the mIOU')
    evaluator = MscEval(net, dl)

    ## evaluate
    mIOU, time_todevice, time_forward, time_max, time_tocpu = evaluator.evaluate()

    best_mIoU = mIOU
    best_iter = 100000

    print('mIOU is: {:.6f}'.format(mIOU))
    print('iter: {:.6f}'.format(best_iter))

    print('best mIOU is: {:.6f}'.format(best_mIoU))
    print('best iter: {:.6f}'.format(best_iter))
    print('Time forward: {:.6f}'.format(time_forward))

    ## find best early stop point
    for mi in range(iter_start, iter_end, 50):
        try:
            net.to('cpu')
            save_pth = osp.join(respth, str(mi)+'_model.pth')
            print(str(mi))
            net.load_state_dict(torch.load(save_pth,map_location='cuda:0'))
            net.cuda()
            net.eval()

            ## evaluator
            print('\n\n\n\n')
            print('compute the mIOU')
            evaluator = MscEval(net, dl)

            ## eval
            mIOU, time_todevice, time_forward, time_max, time_tocpu = evaluator.evaluate()

            if mIOU>best_mIoU:
                best_mIoU = mIOU
                best_iter = mi

            print('mIOU is: {:.6f}'.format(mIOU))
            print('iter: {:.6f}'.format(mi))

            print('best mIOU is: {:.6f}'.format(best_mIoU))
            print('best iter: {:.6f}'.format(best_iter))
        except:
            pass


    print('\n\n\n\n')
    print(net)
    print('find best done, model : {}'.format(respth))
    print('\n\n\n\n')
    print('compute the best mIOU end')
    print('best mIOU is: {:.6f}'.format(best_mIoU))
    print('best iter: {:.6f}'.format(best_iter))
    print('Time forward: {:.6f}'.format(time_forward))



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--explabel',
        dest='explabel',
        type=str,
        default='modelv1',
    )
    parse.add_argument(
        '--expnum',
        dest='expnum',
        type=str,
        default='0926-192138',
    )
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=50000,
    )
    parse.add_argment(
        '--checkpoint_path',
        dest='checkpoint_path',
        type=str,
        default='./results/'
    )
    parse.add_argment(
        '--data_path',
        dest='data_path',
        type=str,
        default='/mnt/hdd1/data/LGE_dataset/'
    )


    args = parse.parse_args()
    findbest(args, args.expnum)

