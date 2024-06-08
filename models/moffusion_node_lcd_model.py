# Reference: The model architecture is modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import os
from collections import OrderedDict
from functools import partial

import numpy as np
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.vqvae_networks.network import VQVAE
from models.networks.diffusion_networks.network import DiffusionUNet
from models.model_utils import load_vqvae

# ldm util
from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    exists,
    default,
)
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

# distributed 
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class MOFFUSIONNodeLCDModel(BaseModel):
    def name(self):
        return 'MOFFUSION-Node-LCD-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.encoders = opt.encoders

        
        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        
        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.unet_params = unet_params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)

        self.df.to(self.device)
        #self.df.cuda()
        self.init_diffusion_params(opt=opt)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)
        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
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

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
            # self.schedulers = [self.scheduler]


        # setup renderer
        dist, elev, azim = 1.7, 20, 20
            
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)

            self.df_module = self.df.module
            self.vqvae_module = self.vqvae.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae

        self.ddim_steps = 200
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, opt=None):
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule()
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
    ############################ END: init diffusion params ############################
    
    def set_input(self, input=None, max_sample=None):
        
        self.x = input['sdf']
        self.lcd = input['lcd']
        self.topo = (input['id'])[:,0]
        self.node = (input['id'])[:,1]
        self.edge = (input['id'])[:,2]


        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.lcd = self.lcd[:max_sample]
            self.topo = self.topo[:max_sample]
            self.node = self.node[:max_sample]
            self.edge = self.edge[:max_sample]

        vars_list = ['x', 'lcd','topo', 'node', 'edge']

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        """
            self.model = DiffusionWrapper(unet_config, conditioning_key)
            => which is the denoising UNet.
        """

        '''
        if isinstance(cond, dict):
            pass
        else:
            #if not isinstance(cond, list):
            #    cond = [cond]
            key = 'c_class' if self.df_module.conditioning_key == 'class' else 'c_crossattn'
            cond = {key: cond}
        '''
        cond = {'c_class':cond}
            

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond1, cond2, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, [cond1, cond2])

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t.to('cpu')].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'loss_total': loss.clone().detach().mean()})

        return x_noisy, target, loss, loss_dict


    def forward(self):

        self.df.train()

        B = self.x.shape[0]

        c_node = self.node
        
        c_lcd   = self.lcd.view(B, 1, 1, 1, 1).repeat(1, self.unet_params.out_channels, self.unet_params.image_size, self.unet_params.image_size, self.unet_params.image_size).float()


        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)
        

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c_node, c_lcd, t)

        self.loss_df = loss
        self.loss_dict = loss_dict

    @torch.no_grad()
    def inference(self, data, ddim_steps=200, ddim_eta=0.):

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        ddim_sampler = DDIMSampler(self)

        # get noise, denoise, and decode with vqvae

        c_node = self.node
        B = c_node.shape[0]
        c_lcd   = self.lcd.view(B, 1, 1, 1, 1).repeat(1, self.unet_params.out_channels, self.unet_params.image_size, self.unet_params.image_size, self.unet_params.image_size).float()
        
        shape = self.z_shape

        
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=[c_node, c_lcd],
                                                     verbose=False,
                                                     eta=ddim_eta)


        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        self.switch_train()


    @torch.no_grad()
    def cond(self, ngen=1, target1='N599', target2=0.30,  ddim_steps=200, ddim_eta=0.):
        ddim_sampler = DDIMSampler(self)
             
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        # get noise, denoise, and decode with vqvae
        B = ngen
        shape = self.z_shape
        target1 = torch.LongTensor(self.encoders['node'].transform([target1]))
        target1 = torch.LongTensor(target1)
        c1 = target1.repeat(B)
        c1 = c1.to(self.device)

        target2 = torch.tensor(target2)
        c2 = target2.repeat(B, self.unet_params.out_channels, self.unet_params.image_size, self.unet_params.image_size, self.unet_params.image_size).float()
        c2 = c2.to(self.device)

        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=[c1, c2],
                                                     verbose=False,
                                                     eta=ddim_eta)


        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        return self.gen_df

    '''
    @torch.no_grad()
    def shape_comp(self, shape, xyz_dict, ngen=1, ddim_steps=100, ddim_eta=0.0):        
        from utils.demo_util import get_partial_shape
        ddim_sampler = DDIMSampler(self)

            
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        if shape.dim() == 4:
            shape = shape.unsqueeze(0)
            shape = shape.to(self.device)
            
        self.df.eval()

        # get noise, denoise, and decode with vqvae
        B = ngen
        z = self.vqvae(shape, forward_no_quant=True, encode_only=True)

        # get partial shape
        ret = get_partial_shape(shape, xyz_dict=xyz_dict, z=z)
        
        x_mask, z_mask = ret['shape_mask'], ret['z_mask']

        # for vis purpose
        self.x_part = ret['shape_part']
        self.x_missing = ret['shape_missing']
        
        shape = self.z_shape
        c = None
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c,
                                                     verbose=False,
                                                     x0=z,
                                                     mask=z_mask,
                                                     eta=ddim_eta)


        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        
        return self.gen_df
    '''
        
    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        self.train()
        return ret

    def backward(self):
        
        self.loss = self.loss_df

        self.loss_dict = reduce_loss_dict(self.loss_dict)
        self.loss_total = self.loss_dict['loss_total']
        self.loss_simple = self.loss_dict['loss_simple']
        self.loss_vlb = self.loss_dict['loss_vlb']
        if 'loss_gamma' in self.loss_dict:
            self.loss_gamma = self.loss_dict['loss_gamma']

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.set_requires_grad([self.df], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.data),
            ('simple', self.loss_simple.data),
            ('vlb', self.loss_vlb.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_sdf(self.renderer, self.gen_df)
            
        vis_tensor_names = [
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'df': self.df_module.state_dict(),
            'opt': self.optimizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()
        
        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        
        
        if self.isTrain:
            self.start_i = state_dict['global_step']
            self.vqvae.load_state_dict(state_dict['vqvae'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


