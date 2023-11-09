import os
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['SVDataLoader']


class SVDataLoader(object):
    r""" PyTorch DataLoader for SV dataset.
    """

    def __init__(self, root, batch_size, num_workers, L, device):
        print(root)
        assert os.path.isdir(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        dir_train = os.path.join(root, f"H{L}_train.mat")
        dir_val = os.path.join(root, f"H{L}_val.mat")
        dir_test = os.path.join(root, f"H{L}_test.mat")
        self.channel, self.nt, self.nc, self.L = 2, 32, 32, L

        # data loading
        Hc_train, Hs_train, Hp_train, Index_train = self._pack(dir_train)
        self.train_dataset = TensorDataset(Hc_train, Hs_train, Hp_train, Index_train)
        Hc_val, Hs_val, Hp_val, Index_val = self._pack(dir_val)
        self.val_dataset = TensorDataset(Hc_val, Hs_val, Hp_val, Index_val)
        Hc_test, Hs_test, Hp_test, Index_test = self._pack(dir_test)
        self.test_dataset = TensorDataset(Hc_test, Hs_test, Hp_test, Index_test)

    def _pack(self, dir):
        matfile = sio.loadmat(dir)
        Hc_A = matfile['HCA']
        Hc_A = torch.tensor(Hc_A, dtype=torch.float32).view(
            Hc_A.shape[0], self.channel, self.nt, self.nc).to(self.device)
        
        Hc_C = matfile['HCC']
        Hc_C = torch.tensor(Hc_C, dtype=torch.float32).view(
            Hc_C.shape[0], self.channel, self.L, self.nc).to(self.device)
        
        Hc_P = matfile['HCP']
        Hc_P = torch.tensor(Hc_P, dtype=torch.float32).view(
            Hc_P.shape[0], self.channel, self.nt, self.nc).to(self.device)
        
        Index = matfile['INDEX']
        Index = torch.tensor(Index, dtype=torch.float32).view(
            Index.shape[0], self.L).to(self.device)
        
        return ([Hc_A, Hc_C, Hc_P, Index])
       

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False)
        print('Data loading FINISH')
        return train_loader, val_loader, test_loader
