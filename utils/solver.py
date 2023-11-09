import time
import os
import torch
from collections import namedtuple
from utils import logger
from utils.statics import AverageMeter, evaluator
import scipy.io as io
__all__ = ['Trainer', 'Tester']


field = ('nmse','epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """
    def __init__(self, mode, model, device, optimizer, train_criterion, test_criterion, scheduler, save_path='./checkpoint', print_freq=20, val_freq=10, test_freq=10):
        
        # Basic arguments
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.test_criterion = test_criterion
        self.scheduler = scheduler
        self.device = device
        
        # Verbose arguments
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.best_nmse = Result()
        
        self.validator = Tester(mode, model, device, test_criterion, print_freq)
        self.tester = Tester(mode, model, device, test_criterion, print_freq)
        
        self.test_loader = None
        self.train_loss = []
        self.val_loss = []
        self.val_nmse = []
        self.test_nmse = []
        
    def loop(self, epochs, train_loader, val_loader, test_loader, save_trend=False):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
            save_trend (bool): whether to save training loss and nmse for visualization.
        """

        self.all_epoch = epochs
        val_nmse = None
        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            train_loss  = self.train(train_loader)
            self.train_loss.append(train_loss)

            if ep % self.val_freq == 0:
                val_loss, val_nmse  = self.val(val_loader)
                self.val_loss.append(val_loss)
                self.val_nmse.append(val_nmse)
                
            if ep % self.test_freq == 0:
                _, test_nmse = self.test(test_loader)
                self.test_nmse.append(test_nmse)
                
            # conduct saving 
            self._loop_postprocessing(val_nmse)
        
        if save_trend == True:    
            io.savemat(os.path.join(self.save_path, f'test_nmse.mat'),{'nmse':torch.tensor(self.test_nmse).numpy()})
            io.savemat(os.path.join(self.save_path, f'train_loss.mat'),{'loss':torch.tensor(self.train_loss).numpy()})
            io.savemat(os.path.join(self.save_path, f'val_loss.mat'),{'loss':torch.tensor(self.val_loss).numpy()})
            io.savemat(os.path.join(self.save_path, f'val_nmse.mat'),{'nmse':torch.tensor(self.val_nmse).numpy()})

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.
        Args:
            train_loader (DataLoader): the training data loader
        """
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" exam the model with validation set.
        Args:
            val_loader: (DataLoader): the validation data loader
        """
        self.model.eval()
        with torch.no_grad():
            val_loss, val_nmse = self.validator(val_loader, verbose=False)
            logger.info(f'=> Val NMSE: {val_nmse:.3e}\n')
            return val_loss, val_nmse

    def test(self, test_loader):
        r""" Truly test the model on the test dataset for one epoch.
        Args:
            test_loader (DataLoader): the test data loader
        """
        self.model.eval()
        with torch.no_grad():
            test_loss, test_nmse = self.tester(test_loader, verbose=False)
            logger.info(f'=> Test NMSE: {test_nmse:.3e}\n')
            return test_loss, test_nmse

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (hca, hcc, hcp, index, ) in enumerate(data_loader):
            if self.mode == 'FB':
                hcc = hcc.to(self.device)
                hcc_pred = self.model(hcc)
                loss = self.train_criterion(hcc_pred, hcc)
            elif self.mode == 'RE':
                hcp = hcp.to(self.device)
                hca_re = self.model(hcp)
                loss = self.train_criterion(hca_re-hcp, hca-hcp)
            elif self.mode == 'Joint':
                hcc = hcc.to(self.device)
                index = index.to(self.device)
                hca_pred = self.model(hcc, index)
                loss = self.train_criterion(hca_pred, hca)
                
            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()                
                self.scheduler.step()
                
            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0] :.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}')

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')
        return iter_loss.avg
    
    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _loop_postprocessing(self, nmse):
        r""" private function which makes loop() function neater.
        """
        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'best_nmse': self.best_nmse
        }

        if nmse is not None:
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse:
                self.best_nmse = Result(nmse=nmse, epoch=self.cur_epoch)
                state['best_nmse'] = self.best_nmse                
                self._save(state, name=f"best_nmse.pth")

        self._save(state, name='last.pth')

        # print current best results
        if self.best_nmse.nmse is not None:
            logger.info(f'\n   Best NMSE: {self.best_nmse.nmse:.3e} '
                        f'\n       epoch: {self.best_nmse.epoch}\n')


class Tester:
    r""" The testing interface for classification
    """
    def __init__(self, mode, model, device, criterion, print_freq=20):
        self.mode = mode
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.
        Args:
            test_data (DataLoader): Data loader for validation data.
        """
        self.model.eval()
        with torch.no_grad():
            loss, nmse = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'  NMSE: {nmse:.3e}\n')
        return loss, nmse

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        iter_nmse = AverageMeter('Iter NMSE')
        time_tmp = time.time()

        for batch_idx, (hca, hcc, hcp, index, ) in enumerate(data_loader):
            if self.mode == 'FB':
                hcc = hcc.to(self.device)
                hcc_pred = self.model(hcc)
                loss = self.criterion(hcc_pred, hcc)
                nmse = evaluator(hcc_pred, hcc)
            elif self.mode == 'RE':
                hcp = hcp.to(self.device)
                hca_re = self.model(hcp)
                loss = self.criterion(hca_re-hcp, hca-hcp)
                nmse = evaluator(hca_re, hca)
            elif self.mode == 'Joint':
                hcc = hcc.to(self.device)
                index = index.to(self.device)
                hca_pred = self.model(hcc, index)
                loss = self.criterion(hca_pred, hca)
                nmse = evaluator(hca_pred, hca)
        
            # Log and visdom update
            iter_loss.update(loss)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')
        return iter_loss.avg, iter_nmse.avg