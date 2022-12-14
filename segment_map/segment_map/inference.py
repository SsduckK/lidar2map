import torch
import argparse
import json
import os.path as osp
import os

import torchvision.transforms as transforms
# from utils.transform import * 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from loadModel import loadmodel


def lb_mapping(lb, lb_map):
    r = lb.copy()
    g = lb.copy()
    b = lb.copy()
    for k, v in lb_map.items():
        r[lb == k] = v[0]*255
        g[lb == k] = v[1]*255
        b[lb == k] = v[2]*255

    rgb = np.zeros((lb.shape[0], lb.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def inference(net,image):

    with open(cfg.LABEL_INFO, 'r') as fr:
        labels_info = json.load(fr)
    lb_map = {el['trainId']: el['color'] for el in labels_info}
    to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.61849181, 0.61981572, 0.62074035), (0.01962486, 0.01949169, 0.01948003)),
    ])

    image = to_tensor(image)
    image = torch.unsqueeze(image,0)
    #
    # image = image.cuda()
    #
    with torch.no_grad():
        prob,_ = net(image,(640, 480),(640,480))
        prod_device = prob.max(1)[1]
        preds = prod_device.data.cpu().numpy()
        label = lb_mapping(preds[0, :, :],lb_map)
        lbimage = Image.fromarray(label.astype(np.uint8))

        
    return lbimage


def show_segmap(image):
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--img_name',
        dest='img_name',
        type=str,
        default= image
    )
    args = parse.parse_args()
    n_classes = 11
    impth = args.img_name

    print('load network\n')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = loadmodel(n_classes)
    net.load_state_dict(torch.load(cfg.WEIGHT,map_location=device),strict=False)
    # net.cuda()
    net.eval()

    # print('load image\n')
    # image = Image.open(impth)

    # args = parse.parse_args()
    result = inference(net, image)
    return result

if __name__ == "__main__":
    show_segmap()