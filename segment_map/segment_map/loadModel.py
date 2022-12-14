

from model.swiftnet.models.semseg import SemsegModel
from model.swiftnet.models.resnet.resnet_single_scale import resnet18

def loadmodel(n_classes):
    
    resnet = resnet18(pretrained=True, efficient=False)
    net = SemsegModel(resnet, n_classes)

    return net
