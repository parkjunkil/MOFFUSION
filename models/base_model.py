# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import os
from termcolor import colored, cprint
import torch
import utils.util as util

def create_model(opt):
    model = None

    if opt.model == 'vqvae_moffusion':
        from models.vqvae_model import VQVAEModel
        model = VQVAEModel()
    
    elif opt.model == 'mof_constructor_topo':
        from models.mof_constructor_topo_model import ResNetModel
        model = ResNetModel()

    elif opt.model == 'mof_constructor_BB':
        from models.mof_constructor_BB_model import ResNetModel
        model = ResNetModel()

    elif opt.model == 'moffusion_uncond':
        from models.moffusion_model import MOFFUSIONModel
        model = MOFFUSIONModel()

    elif opt.model == 'moffusion_H2':
        from models.moffusion_H2_model import MOFFUSIONH2Model
        model = MOFFUSIONH2Model()

    elif opt.model == 'moffusion_topo':
        from models.moffusion_topo_model import MOFFUSIONTopoModel
        model = MOFFUSIONTopoModel()

    elif opt.model == 'moffusion_node_lcd':
        from models.moffusion_node_lcd_model import MOFFUSIONNodeLCDModel
        model = MOFFUSIONNodeLCDModel()
        
    elif opt.model == 'moffusion_vf_sa':
        from models.moffusion_vf_sa_model import MOFFUSIONVfSaModel
        model = MOFFUSIONVfSaModel()

    elif opt.model == 'moffusion_text':
        from models.moffusion_text_model import MOFFUSIONTextModel
        model = MOFFUSIONTextModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    cprint("[*] Model has been created: %s" % model.name(), 'blue')
    return model

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass
        
    def get_current_errors(self):
        return {}

    # define the optimizers
    def set_optimizers(self):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # update learning rate (called once every epoch)
    def update_learning_rate(self, loss):
        for scheduler in self.schedulers:
            scheduler.step(loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('[*] learning rate = %.7f' % lr)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                # setattr(self, name, var.cuda(self.gpu_ids[0], non_blocking=True))
                setattr(self, name, var.cuda(self.opt.device, non_blocking=True))


    def tnsrs2ims(self, tensor_names):
        ims = []
        for name in tensor_names:
            if isinstance(name, str):
                var = getattr(self, name)
                ims.append(util.tensor2im(var.data))
        return ims
