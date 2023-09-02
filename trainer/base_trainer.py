import os
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint
import pathlib
from torch.utils.tensorboard import SummaryWriter
from logger import init_logger
import pprint

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_set, valid_set, model, losses, args):
        self.train_set = train_set
        self.valid_set = valid_set
        self.args = args
        self.model = model
        self.optimizer = self._get_optimizer()
        self.best_error = np.inf
        self.save_root = pathlib.Path(self.args.save_root)
        self.i_epoch = self.args.after_epoch+1
        self.i_iter = 0
        self.model_suffix = args.model_suffix
        self.loss_modules = losses
        #self.loss_module = losses['loss_module']
        #if 'mwl_module' in losses.keys():
        #    self.mwl_module = losses['mwl_module']
        #if 'cyc_module' in losses.keys():
        #    self.cyc_module = losses['cyc_module']

    def train(self, rank, world_size):
        self._init_rank(rank,world_size)

        for l_idx, epochs in enumerate(self.args.levels):
            if "ncc" in self.args.loss:
                self.loss_modules['loss_module'].ncc_win = self.args.ncc_win[l_idx]
                self.loss_modules['loss_module'].w_ncc_scales = self.args.w_ncc_scales[l_idx]
                self.epochs = epochs

            for epoch in range(epochs):
                self._run_one_epoch()
                
                if self.rank == 0 and self.i_epoch % self.args.valid_interval == 0:
                    errors, error_names = self._validate()

                
                # In order to reduce the space occupied during debugging,
            # only the model with more than cfg.save_iter iterations will be saved.
                if self.args.epoch_size > 0 and self.i_iter > self.args.save_iter:
                    self.save_model(self.loss, name=self.model_suffix)
                self.i_epoch += 1
                torch.cuda.empty_cache()
                
        
        self.cleanup()
        pass

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    def _init_rank(self, rank, world_size):
        self.setup(rank, world_size)
        self.world_size = world_size
        self.rank = rank
        print('DDP: Rank {} initialized'.format(rank))

        # init logger
        if self.rank == 0:
            self._log = init_logger(log_dir=self.args.save_root, filename=self.args.model_suffix + '.log')
            self._log.info('=> Rank {}: will save everything to {}'.format(self.rank, self.args.save_root))

            # show configurations
            cfg_str = pprint.pformat(self.args)
            self._log.info('=> configurations \n ' + cfg_str)
            self._log.info('{} training samples found'.format(len(self.train_set)))
            self._log.info('{} validation samples found'.format(len(self.valid_set)))
            self.summary_writer = SummaryWriter(str(self.args.save_root))
        
        self.train_loader, self.valid_loader = self._get_dataloaders(self.train_set, self.valid_set)
        self.args.epoch_size = min(self.args.epoch_size, len(self.train_loader))

        if self.rank != 0: #! GPU
            torch.cuda.set_device(self.rank)
        self.loss_modules = {loss_: module_.to(self.rank) for loss_, module_ in self.loss_modules.items()}
        #self.loss_module = self.loss_module.to(self.rank)
        #if hasattr(self, 'mwl_module'):
        #    self.mwl_module = self.mwl_module.to(self.rank)
        #if hasattr(self, 'cyc_module'):
        #    self.cyc_module = self.cyc_module.to(self.rank)

        self.model = self._init_model(self.model)
        pass

    def _get_dataloaders(self,train_set,valid_set):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	                    train_set,
                            shuffle=True,
    	                    num_replicas = self.args.n_gpu,
    	                    rank=self.rank)
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_set,
                            batch_size=self.args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(
                            dataset=valid_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)                        
                        
        return train_loader, valid_loader

    def _get_optimizer(self):
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': 0},
            {'params': weight_parameters(self.model),
             'weight_decay': 1e-6}]

        return torch.optim.Adam(param_groups, self.args.lr,
                                betas=(0.9, 0.999), eps=1e-7)

    def _init_model(self, model):
        model = model.to(self.rank)
        if self.args.load:
            if self.rank == 0:
                self._log.info(f'Loading model from {self.args.load}')
            epoch, weights = load_checkpoint(self.args.load)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)

        else:
            if self.rank == 0:
                self._log.info("=> Train from scratch")
            model.apply(model.init_weights)

        if self.world_size > 1:
            if self.rank == 0:
                self._log.info(f'DDP Wrapping the model')
            model = DDP(model, device_ids=[self.rank])
            #model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            # model = DDP(model, device_ids=self.device_ids)

        return model

    def _init_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("There\'s no GPU available on this machine,"
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        #list_ids = list(range(n_gpu_use))
        return device, n_gpu_use

    def save_model(self, error, name):
        is_best = error < self.best_error        
        #! it will try to save only if we improved the loss!
        if is_best:
            self.best_error = error
            
            try:
                models = {'epoch': self.i_epoch, 'state_dict': self.model.module.state_dict()}
            except:
                models = {'epoch': self.i_epoch, 'state_dict': self.model.state_dict()}
            
            save_checkpoint(self.save_root, models, name, is_best)
    
    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group 
        #! GPUUUUUUU 
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()
