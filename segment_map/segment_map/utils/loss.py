#!/usr/bin/python
# -*- encoding: utf-8 -*-


from os import lseek
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional
# import cv2
import numpy as np
from torchvision.utils import save_image

# class EdgeLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(EdgeLoss, self).__init__()
#         self.criteria = nn.MSELoss(reduction='none')

#     def MakeEdge(self,x):  



#     def forward(self, recon_depth, label):
        
#         loss = self.criteria(probs,depth)
#         mask = F.interpolate(mask, depth.size()[2:], mode='nearest')
#         loss[mask==0] = 0
#         # import pdb
#         # pdb.set_trace()

#         return torch.mean(loss)
# class RelativeLoss(nn.module):
#     def __init__(self, *args, **kwargs):
#         super(RelativeLoss, self).__init__()
#     def forward(self,)


class DepthMaskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DepthMaskLoss, self).__init__()
        self.criteria = nn.MSELoss(reduction='none')
    def forward(self, probs, depth,mask):  
        mask = F.interpolate(mask, depth.size()[2:], mode='nearest').cuda()
        # depth = torch.mul(depth,mask)
        # probs = torch.mul(probs,mask)
        # # depth[mask==0] = 0
        # # probs[mask==0] = 0
        # loss = self.criteria(probs,depth)

        # diff2 = (torch.flatten(probs) - torch.flatten(depth)) ** 2.0
        # sum2 = 0.0
        # num = 0

        # flat_mask = torch.flatten(mask)
        # assert(len(flat_mask) == len(diff2))
        # for i in range(len(diff2)):
        #     if flat_mask[i] != 0:
        #         sum2 += diff2[i]
        #         num += 1

        # out = torch.sum(((probs-depth)*mask)**2.0)  / torch.sum(mask)
        # return out

        diff2 = (torch.flatten(depth) - torch.flatten(probs)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class DepthMaskLoss_5ch(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DepthMaskLoss_5ch, self).__init__()
        self.criteria = nn.MSELoss(reduction='none')

    def forward(self, probs, depth, mask):
        mask = F.interpolate(mask, depth.size()[2:], mode='nearest').cuda()
        depth = torch.mul(depth,mask)
        probs = torch.mul(probs,mask)


        loss = self.criteria(probs,depth)
        # loss[mask==0] = 0  
        return torch.sum(loss)/mask.sum()

class ProbShifting(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ProbShifting, self).__init__()
        
    def forward(self, probs):
        padded_probs = F.pad(input = probs, mode = 'replicate',pad=(1,1,1,1))

        d_size = probs.size(3)
        d_size_n = probs.size(2)
        left_shift = padded_probs[:,:,2:(d_size_n+2),1:(d_size+1)]
        right_shift = padded_probs[:,:,0:(d_size_n),1:(d_size+1)]
        down_shift = padded_probs[:,:,1:(d_size_n+1),2:(d_size+2)]
        up_shift = padded_probs[:,:,1:(d_size_n+1),0:(d_size)]
    

        l= abs(left_shift-probs)
        r= abs(right_shift-probs)
        d= abs(down_shift-probs)
        u= abs(up_shift-probs)       

        return l,r,d,u


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        # import pdb
        # pdb.set_trace()
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class RotationLoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.MSELoss(reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feator, feat_rot):
        
        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0

        feator = torch.mul(feator,mask)

        loss = self.criteria(feator, feat_rot).view(-1)
 
        return torch.mean(loss)

class RotationL1Loss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationL1Loss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.L1Loss(reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feator, feat_rot):
        
        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0

        feator = torch.mul(feator,mask)

        loss = self.criteria(feator, feat_rot).view(-1)
 
        return torch.mean(loss)

class RotationCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feat_rot, label ):
        
        N, C, H, W = feat_rot.size()

        mask = torch.ones([N,C,H,W])
        mask_zero = torch.zeros([N,C,H,W])

        mask = mask.to(self.device)
        mask_zero = mask_zero.to(self.device)
        
        mask[feat_rot == 0 ] = 0
        mask_zero[feat_rot == 0] = 255

        mask = mask[:,0,:,:]
        mask_zero = mask_zero[:,0,:,:]
        
        label = torch.mul(label,mask)
        label = label + mask_zero
        label = label.type(torch.LongTensor)
        label = label.to(self.device)

        loss = self.criteria( feat_rot ,label)
 
        return torch.mean(loss)


class  OhemCELossRot(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELossRot, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, logits, labels, feat):
        # import pdb
        # pdb.set_trace()
        N, C, H, W = feat.size()
        N1, C1, H1, W1 = logits.size() 

        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)

        mask[feat == 0 ] = 0
        mask = mask[:,0:C1,:,:]
        
        
        mask = F.interpolate(mask, [H1,W1], mode='nearest')
        mask1 = mask[:,0,:,:]

        # labels = torch.mul(labels,mask1)

        # labels_ = labels
        
        # labels_[mask1 == 0] = 15

        # logits = torch.mul(logits,mask)
        # labels.type(torch.LongTensor)

        loss = self.criteria(logits, labels)
        # .view(-1)

        loss = torch.mul(loss,mask1).view(-1)

        loss, _ = torch.sort(loss, descending=True)

        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class RotationGAPLoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationGAPLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.L1Loss(reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, feator, feat_rot):
        
        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0

        feator = torch.mul(feator,mask)

        dist = torch.sub(feator,feat_rot)

        avgpool = nn.AvgPool2d(dist.size()[2:],stride=1)
        diff = avgpool(dist)

        med = torch.median(diff)

        shape_mask = torch.ones(diff.size())
        shape_mask[diff>med] = 0
        shape_mask = shape_mask.to(self.device)

        feator= torch.mul(feator,shape_mask)
        feat_rot = torch.mul(feat_rot,shape_mask)       

        loss = self.criteria(feator, feat_rot).view(-1)

        # n_min = int(9077760*0.4)

        # loss, _ = torch.sort(loss, descending=True)

        # loss = loss[n_min:] #Bottom N
        # # loss = loss[:n_min] # Top N

 
        return torch.mean(loss)


class RotationL1FocalLoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationL1FocalLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.L1Loss(reduction='none')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feator, feat_rot):

# ##################################################################

#         N, C, H, W = logits.size()
#         loss = self.criteria(logits, labels).view(-1)
#         loss, _ = torch.sort(loss, descending=True)
#         if loss[self.n_min] > self.thresh:
#             loss = loss[loss>self.thresh]
#         else:
#             loss = loss[:self.n_min]
    
# ##################################################################

        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0

        feator = torch.mul(feator,mask)

        loss = self.criteria(feator, feat_rot).view(-1)
        
        # loss = self.criteria(feator, feat_rot)
        n_min = int(9077760*0.4)

        loss, _ = torch.sort(loss, descending=True)

        loss = loss[n_min:] #Bottom N
        # loss = loss[:n_min] # Top N

 
        return torch.mean(loss)

class RotationPDLoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationPDLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.PairwiseDistance(p=2.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feator, feat_rot):
        
        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0

        feator = torch.mul(feator,mask)

        loss = self.criteria(feator, feat_rot).view(-1)
 
        return torch.mean(loss)


class RotationCOSLoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(RotationCOSLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        # self.sim = F.cosine_similarity(dim=1, eps=1e-08)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feator, feat_rot):
        
        N, C, H, W = feator.size()
        
        mask = torch.ones([N,C,H,W])
        mask = mask.to(self.device)
        mask[feat_rot == 0 ] = 0
       
        feator = torch.mul(feator,mask)

        or_min = torch.min(feator)
        or_max = torch.max(feator)
        rot_min = torch.min(feat_rot)
        rot_max = torch.min(feat_rot)

        if or_min>rot_min:
            feator = torch.sub(feator,rot_min)
            feat_rot = torch.sub(feat_rot,rot_min)
        else:
            feator = torch.sub(feator,or_min)
            feat_rot = torch.sub(feat_rot,or_min)

        if or_max>rot_max:
            feator = torch.div(feator,or_max)
            feat_rot = torch.div(feat_rot,or_max)
        else:
            feator = torch.div(feator,rot_max)
            feat_rot = torch.div(feat_rot,rot_max)

        loss = 1-F.cosine_similarity(feator,feat_rot)

 
        return torch.mean(loss)

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
