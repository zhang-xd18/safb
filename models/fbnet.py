import torch
import torch.nn as nn
import sys
from collections import OrderedDict
sys.path.append("..")
from utils import logger

__all__ = ['fbnet']

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


class FEncoder(nn.Module):
    def __init__(self):
        super(FEncoder, self).__init__()        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv1x3_bn", ConvBN(2, 7, [1, 3])), 
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),            
            ("conv1x9_bn(1)", ConvBN(7, 7, [1, 9])), 
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),                   
            ("conv1x9_bn(2)", ConvBN(7, 7, [1, 9])),
        ]))
        self.relu1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
    
    
        self.encoder2 = nn.Sequential(OrderedDict([
            ("conv1x3_bn(1)", ConvBN(2, 7, [1, 3])),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x3_bn(2)", ConvBN(7, 7, [1, 3])),
        ]))
        self.relu2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.encoder_conv = nn.Sequential(OrderedDict([
            ("conv1x1_bn", ConvBN(7 * 2, 2, 1)), 
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True))
        ]))
        
    def forward(self, x):
        encode1 = self.encoder1(x)
        out1 = self.relu1(encode1)
        
        encode2 = self.encoder2(x)
        out2 = self.relu2(encode2)
        
        out = torch.cat((out1, out2), dim=1)
        out = self.encoder_conv(out)
        return out

class FBlock(nn.Module):
    def __init__(self):
        super(FBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x3_bn', ConvBN(2, 7, [1, 3])), 
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9_bn', ConvBN(7, 7, [1, 9])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5_bn', ConvBN(2, 7, [1, 5])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)


    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out    

class FBNet(nn.Module):
    def __init__(self, L=5, reduction=4):
        super(FBNet, self).__init__()
        in_channel, h, w = 2, L, 32
        real_size = in_channel * h * w
        total_size = in_channel * 32 * w
        logger.info(f'reduction={reduction}')
        
        self.encoder_feature = nn.Sequential(OrderedDict([
            ("encoder", FEncoder()),
        ]))
        
        self.encoder_fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(real_size, total_size // reduction)),
        ]))
        
        self.decoder_fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(total_size // reduction, 4 * total_size // reduction)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('linear2', nn.Linear(4 * total_size // reduction, real_size)),
        ]))        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv3x5_bn", ConvBN(2, 2, [3, 5])),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("FBlock1", FBlock()),
            ("FBlock2", FBlock()),
            ("FBlock3", FBlock())
        ]))
        
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hcc):
        n, c, h, w = hcc.detach().size()
        
        out = self.encoder_feature(hcc)
        out = self.encoder_fc(out.view(n,-1))
        
        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out)
        
        out = self.sigmoid(out)
        return out 

def fbnet(L=5, reduction=4):
    r""" Create a proposed FBNet.
    :param reduction: the reciprocal of compression ratio
    :return: an instance of FBNet
    """

    model = FBNet(L=L, reduction=reduction)
    return model
