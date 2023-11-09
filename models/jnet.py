import torch
import torch.nn as nn
from .fbnet import fbnet
from .renet import renet

__all__ = ["jnet"]

class JNet(nn.Module):
    def __init__(self, L=5, reduction=4):
        super(JNet, self).__init__()
        self.FBNet = fbnet(L=L, reduction=reduction)
        self.RENet = renet()
    
    def forward(self, hcc, index):
        
        # # Just for test
        # n, c, h, w = hcc.detach().size()
        # hc=0
        # if h < 10:
        #     hc = 10 - h
        # hcc = torch.cat((hcc, torch.zeros([n,c,hc,w], device=hcc.device) + 0.5), dim=2)
        
        hcc_out = self.FBNet(hcc)
        
        # # Just for test
        # hcc_out = hcc_out[:,:,0:h,:]
        
        hcp_out = self._mapping(hcc_out, index)
        out = self.RENet(hcp_out)
        return out
        
        
    def _mapping(self, hcc, index):
        n, c, h, w = hcc.detach().size()
        hcp_out = torch.zeros([n, 1, h, w], device=hcc.device)
        
        assist = hcp_out.scatter(3,(index-1).long().unsqueeze(1).unsqueeze(3), 1)
        hcp_out = torch.matmul(assist.transpose(2,3), hcc - 0.5) + 0.5
        
        return hcp_out
        

def jnet(L=5, reduction=4):
    r""" Create a proposed JNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of JNet
    """

    model = JNet(L=L, reduction=reduction)
    return model