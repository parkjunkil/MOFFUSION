# Reference: The model architecture is modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import os
from collections import OrderedDict
from functools import partial
from inspect import isfunction

import cv2
import numpy as np
import einops
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
from models.networks.bert_networks.scibert_network import create_scibert
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

class MOFFUSIONTextModel(BaseModel):
    def name(self):
        return 'MOFFUSION-Text-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

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

        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)
        self.df.to(self.device)
        self.init_diffusion_params(opt=opt)
        
        # sampler 
        self.ddim_sampler = DDIMSampler(self)
        
        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)

        # init cond model
        bert_params = df_conf.bert.params
        self.text_embed_dim = bert_params.n_embed
        self.cond_model, self.cond_model_tokenizer = create_scibert()
        self.cond_model.to(self.device)

        
        for param in self.cond_model.parameters(): 
            param.requires_grad = False # No grad on conditional model
        
        ######## END: Define Networks ########
        self.cond_layer = nn.Linear(self.cond_model.config.hidden_size, self.cond_model.config.hidden_size)
        self.cond_layer.to(self.device)


        # param list
        trainable_models = [self.df, self.cond_model]
        trainable_params = []
        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]
            # print(len(trainable_params))

        if self.isTrain:
            
            # initialize optimizers
            self.optimizer = optim.AdamW(trainable_params, lr=opt.lr)
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

        # transforms
        self.to_tensor = transforms.ToTensor()

        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for multi-gpu
        if self.opt.distributed:
            self.make_distributed(opt)

            self.df_module = self.df.module
            self.vqvae_module = self.vqvae.module
            self.cond_model_module = self.cond_model.module
            self.cond_layer_module = self.cond_layer.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae
            self.cond_model_module = self.cond_model
            self.cond_layer_module = self.cond_layer

        # for debugging purpose
        self.ddim_steps = 100
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
        self.cond_model = nn.parallel.DistributedDataParallel(
            self.cond_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.cond_layer = nn.parallel.DistributedDataParallel(
            self.cond_layer,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, opt=None):
        
        df_conf = OmegaConf.load(opt.df_cfg)
        df_model_params = df_conf.model.params
        
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule(
            timesteps=df_model_params.timesteps,
            linear_start=df_model_params.linear_start,
            linear_end=df_model_params.linear_end,
        )
        
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)

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
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
        ############################ END: init diffusion params ############################

    def set_input(self, input=None, max_sample=None):

        self.x = input['sdf']
        self.text = input['text']
        encoded_input = self.cond_model_tokenizer(self.text, return_tensors='pt', padding="max_length", max_length=256)
        self.input_ids = encoded_input['input_ids']
        self.attention_mask = encoded_input['attention_mask']


        B = self.x.shape[0]
        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.text = self.text[:max_sample]
            #self.uc_text = self.uc_text[:max_sample]

        vars_list = ['x',
                     'input_ids', 'attention_mask',]
        #            'uc_input_ids', 'uc_attention_mask']


        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()
        self.cond_model.train()
        self.cond_layer.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.cond_model.eval()
        self.cond_layer.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

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


    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, cond)

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

        logvar_t = self.logvar[t].to(self.device)
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

        self.switch_train()

        with torch.no_grad():
            c = self.cond_model(input_ids=self.input_ids,
                                attention_mask=self.attention_mask)['last_hidden_state']
        c = self.cond_layer(c)
        # c_backbone = c_backbone['encoder_out'][0].permute(1, 0, 2)
        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        #    check: ldm.models.autoencoder.py, VQModelInterface's encode(self, x)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c, t)

        self.loss_df = loss
        self.loss_dict = loss_dict

    @torch.no_grad()
    def inference(self, data, ddim_steps=200, ddim_eta=0.0, infer_all=False, max_sample=16):

        self.switch_eval()

        if not infer_all:
            # max_sample = 16
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        
        ddim_sampler = DDIMSampler(self)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        # get noise, denoise, and decode with vqvae
        c = self.cond_model(input_ids=self.input_ids,
                            attention_mask=self.attention_mask)['last_hidden_state']
        c = self.cond_layer(c)

        B = c.shape[0]
        shape = self.z_shape
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c,
                                                     verbose=False,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        self.switch_train()


    @torch.no_grad()
    def txt2shape(self, input_txt, ngen=6, ddim_steps=200, ddim_eta=0.0):

        self.switch_eval()

        data = {
            'sdf': torch.zeros(ngen),
            'text': [input_txt] * ngen,
        }
        
        self.set_input(data)

        ddim_sampler = DDIMSampler(self)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        # get noise, denoise, and decode with vqvae
        c = self.cond_model(input_ids=self.input_ids,
                            attention_mask=self.attention_mask)['last_hidden_state']
        c = self.cond_layer(c)
        B = c.shape[0]

        shape = self.z_shape
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c,
                                                     verbose=False,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        return self.gen_df

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        
        self.switch_eval()
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
        self.set_requires_grad([self.cond_model], requires_grad=True)
        self.set_requires_grad([self.cond_layer], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('simple', self.loss_simple.mean().data),
            ('vlb', self.loss_vlb.mean().data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.mean().data

        return ret

    def write_text_on_img(self, text, bs=16, img_shape=(3, 256, 256)):
        # write text as img
        b, c, h, w = len(text), 3, 256, 256
        img_text = np.ones((b, h, w, 3)).astype(np.float32) * 255
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_COMPLEX
        font_size = 0.5
        n_char_per_line = 25 # new line for text

        y0, dy = 20, 1
        for ix, txt in enumerate(text):
            # newline every "space" chars
            for i in range(0, len(txt), n_char_per_line):
                y = y0 + i * dy
                # new_txt.append(' '.join(words[i:i+space]))
                # txt_i = ' '.join(txt[i:i+space])
                txt_i = txt[i:i+n_char_per_line]
                cv2.putText(img_text[ix], txt_i, (10, y), font, font_size, (0., 0., 0.), 2)

        return img_text/255.

    
    def get_current_visuals(self):


        with torch.no_grad():
            self.text = self.text # input text
            #self.img_gt = render_sdf(self.renderer, self.x).detach().cpu() # rendered gt sdf
            self.img_gen_df = render_sdf(self.renderer, self.gen_df).detach().cpu() # rendered generated sdf
        
        b, c, h, w = self.img_gen_df.shape
        img_shape = (3, h, w)
        # write text as img
        self.img_text = self.write_text_on_img(self.text, bs=b, img_shape=img_shape)
        self.img_text = rearrange(torch.from_numpy(self.img_text), 'b h w c -> b c h w')

        vis_tensor_names = [
            # 'img',
            'img_gen_df',
            'img_text',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    '''
    def get_current_visuals(self):

        with torch.no_grad():
            self.text = self.text # input text
            self.img_gt = render_sdf(self.renderer, self.x).detach().cpu() # rendered gt sdf
            self.img_gen_df = render_sdf(self.renderer, self.gen_df).detach().cpu() # rendered generated sdf
        
        b, c, h, w = self.img_gt.shape
        img_shape = (3, h, w)
        # write text as img
        self.img_text = self.write_text_on_img(self.text, bs=b, img_shape=img_shape)
        self.img_text = rearrange(torch.from_numpy(self.img_text), 'b h w c -> b c h w')

        vis_tensor_names = [
            # 'img',
            'img_gt',
            'img_gen_df',
            'img_text',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)
    '''
        
    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'df': self.df_module.state_dict(),
            'cond_model': self.cond_model_module.state_dict(),
            'cond_layer': self.cond_layer_module.state_dict(),
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
        self.cond_model.load_state_dict(state_dict['cond_model'])
        self.cond_layer.load_state_dict(state_dict['cond_layer'])
        
        
        if self.isTrain:
            self.start_i = state_dict['global_step']
            self.vqvae.load_state_dict(state_dict['vqvae'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))