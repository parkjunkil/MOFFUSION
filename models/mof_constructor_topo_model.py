# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import os
from collections import OrderedDict

import numpy as np
import mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.profiler import record_function

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.resnet_topo_networks.network import _resnet, BasicBlock
from models.losses import VQLoss

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf
from utils.distributed import reduce_loss_dict

import torch.nn.functional as F


class ResNetModel(BaseModel):
    def name(self):
        return 'MOF-Constructor_Topo-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        # model
        assert opt.res_topo_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.res_topo_cfg)
        mparam = configs.model.params
        ddconfig = mparam.ddconfig

        in_channels = ddconfig.in_channels
        layers = mparam.layers
        kernel = mparam.kernel
        padding = mparam.padding
  
        encoders = opt.encoders
        topo_dim  = len(encoders['topo'].classes_)

        self.resnet = _resnet('resnet18', BasicBlock, layers, in_channels, kernel, padding, topo_dim)
        self.resnet.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.Adam(self.resnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
                                                        patience=10000, threshold=0.0001,
                                                        threshold_mode='rel',
                                                        cooldown=0,
                                                        min_lr=0,
                                                        eps=1e-08,
                                                        verbose=False)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # continue training
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        # for saving best ckpt
        self.best_accuracy = -1e12
        

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.resnet_module = self.resnet.module
        else:
            self.resnet_module = self.resnet

    def switch_eval(self):
        self.resnet.eval()
        
    def switch_train(self):
        self.resnet.train()

    def make_distributed(self, opt):
        self.resnet = nn.parallel.DistributedDataParallel(
            self.resnet,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    def set_input(self, input, max_sample=None):
        
        self.x = input['sdf']

        self.y = input['id']

        self.topo = (input['id'])[:,0]

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.y = self.y[:max_sample]
            self.topo = self.topo[:max_sample]

        vars_list = ['x', 'y', 'topo']
        self.cur_bs = self.x.shape[0] # to handle last batch
        #vars_list = ['x']

        self.tocuda(var_names=vars_list)

    def forward(self):
        y_topo = self.resnet(self.x)

        y_topo_gt = self.topo.clone().detach()

        loss = self.criterion(y_topo, y_topo_gt)

        self.loss = loss

    @torch.no_grad()
    def inference(self, data, verbose=False):
        self.switch_eval()
        self.set_input(data)

        with torch.no_grad():
            self.y_pred = self.resnet(self.x)

            topo_pred = self.y_pred

            _, topo_pred  = (F.softmax(topo_pred, dim=-1)).topk(1)

            y_pred = [topo_pred]
            y_pred = torch.transpose(torch.stack(y_pred),0,1)
            y_pred = y_pred.view(len(data['sdf']),1)

            self.y_pred = y_pred

        self.switch_train()

    def eval_metrics(self, dataloader,  global_step=0):
        self.eval()

        topo_correct  = 0

        tot_len = 0

        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):

                tot_len += len(test_data['sdf'])
            
                self.inference(test_data)

                Y_gt = self.y.clone().detach()

                Y_pred = self.y_pred.detach()

                for i in range(len(Y_gt)):
                    if Y_gt[i][0] == Y_pred[i][0]:
                        print(Y_gt[i][0], Y_pred[i][0])
                        topo_correct += 1


        topo_correct /= tot_len

        ret = OrderedDict([
            ('topo_accuracy', topo_correct),
        ])
        # check whether to save best epoch
        if ret['topo_accuracy'] > self.best_accuracy:
            self.best_accuracy = ret['topo_accuracy']
            save_name = f'epoch-best'
            self.save(save_name, global_step) # pass 0 just now

        self.switch_train()
        return ret


    # only for demo
    def predict(self, df):
        
        self.eval()
        topo_pred = self.resnet(df)
        _, topo_pred = (F.softmax(topo_pred, dim=-1)).topk(1)
        
        return topo_pred 



    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        
        log = {
            "loss_total": self.loss.clone().detach().mean()
        }

        self.loss_dict = reduce_loss_dict(log)

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
    
    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss.mean().data)
        ])

        return ret

    def save(self, label, global_step=0, save_opt=False):

        state_dict = {
            'resnet': self.resnet_module.state_dict(),
            # 'opt': self.optimizer.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'resnet_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        # NOTE: handle version difference...
        if 'resnet' not in state_dict:
            self.resnet.load_state_dict(state_dict)
        else:
            self.resnet.load_state_dict(state_dict['resnet'])
            
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


