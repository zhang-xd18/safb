import torch
import torch.nn as nn
import sys
from collections import OrderedDict
sys.path.append("..")

__all__ = ["renet"]

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class REBlock(nn.Module):
    def __init__(self):
        super(REBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d([3, 1], stride=1, padding=[1, 0])),
            ('conv7x1_bn', ConvBN(2, 7, [7, 1])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_bn(1)', ConvBN(7, 7, [5, 1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_bn(2)', ConvBN(7, 2, [5, 1])),
        ]))
        
        self.path2 = nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d([3, 1], stride=1, padding=[1, 0])),
            ('conv11x1_bn', ConvBN(2, 7, [11, 1])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1_bn(1)', ConvBN(7, 7, [9, 1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1_bn(2)', ConvBN(7, 2, [9, 1])),
        ]))
        
        self.conv = nn.Sequential(OrderedDict([
            ('conv1x1', ConvBN(4, 2, 1)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = self.conv(torch.cat([out1, out2], dim=1))
        return out


class RENet(nn.Module):
    def __init__(self):
        super(RENet, self).__init__()
        self.identity = nn.Identity()
        self.feature = nn.Sequential(OrderedDict([
            ("REBlock", REBlock()),
        ]))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hcp):
        identity = self.identity(hcp)
        
        out = self.feature(hcp - 0.5) + 0.5 # eliminate the bias to accelarate convergence
        
        # generate the index of selected channel columns directly, can also be replaced by using dataset index
        mask = (abs(hcp - 0.5) > 1e-7)  
        
        # only the eliminated columns are complemented by RENet
        out = out - out * mask
        out = identity * mask + out 
        return out 

def renet():
    model = RENet()
    return model