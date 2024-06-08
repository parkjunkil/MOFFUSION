# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import torch
import torchvision.utils as vutils

from datasets.base_dataset import CreateDataset
from datasets.dataloader import CreateDataLoader, get_data_generator

from models.base_model import create_model

from utils.util import seed_everything

from sklearn.preprocessing import LabelEncoder
import joblib
from collections import OrderedDict

def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

############ START: all Opt classes ############

class BaseOpt(object):
    def __init__(self, gpu_ids=0, seed=None):
        # important args
        self.isTrain = False
        self.gpu_ids = [gpu_ids]
        # self.device = f'cuda:{gpu_ids}'
        self.device = 'cuda'
        self.debug = '0'

        # default args
        self.serial_batches = False
        self.nThreads = 4
        self.distributed = False

        # hyperparams
        self.batch_size = 1

        # dataset args
        self.max_dataset_size = 10000000
        self.trunc_thres = 0.2

        if seed is not None:
            seed_everything(seed)
            
        self.phase = 'test'


        self.vq_model = 'vqvae'
        self.vq_cfg = './configs/vqvae.yaml'
        self.vq_ckpt ='./saved_ckpt/vqvae.pth'

        self.res_topo_model = 'mof_constructor_topo'
        self.res_topo_cfg = './configs/mof_constructor.yaml'
        self.res_topo_ckpt = './saved_ckpt/mof_constructor_topo.pth'

        self.res_BB_model = 'mof_constructor_BB'
        self.res_BB_cfg = './configs/mof_constructor.yaml'
        self.res_BB_ckpt = './saved_ckpt/mof_constructor_BB.pth'

        self.topo_enc='./saved_encoders/topo_encoder.joblib'
        self.node_enc='./saved_encoders/node_encoder.joblib'
        self.edge_enc='./saved_encoders/edge_encoder.joblib'
        
        topo_encoder = LabelEncoder()
        node_encoder = LabelEncoder()
        edge_encoder = LabelEncoder()

        topo_encoder = joblib.load(self.topo_enc)
        node_encoder = joblib.load(self.node_enc)
        edge_encoder = joblib.load(self.edge_enc)

        encoders = OrderedDict()
        encoders['topo'] = topo_encoder
        encoders['node'] = node_encoder
        encoders['edge'] = edge_encoder

        self.encoders = encoders

    def name(self):

        return 'BaseOpt'

class VQVAEOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids)

        # some other custom args here

        print(f'[*] {self.name()} initialized.')

    def name(self):
        return 'VQVAETestOpt'

class MOFFUSIONOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        
        # opt.res = 128
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='mof_250k', res=32):
        self.dataroot = dataroot
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode

    def init_model_args(
            self,
            ckpt_path="./saved_ckpt/moffusion_uncond.pth",
            vq_ckpt_path="./saved_ckpt/vqvae.pth",
        ):
        self.model = 'moffusion_uncond'
        self.df_cfg = './configs/moffusion-uncond.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae_moffusion'
        self.vq_cfg = './configs/vqvae.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'mof_250k'

    def name(self):
        return 'MOFFUSIONTestOption'


class MOFFUSIONH2Opt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='mof_250k', res=32):
        # dataset - snet
        self.dataroot = dataroot
        self.res = res
        self.dataset_mode = dataset_mode
        
    def init_model_args(
            self,
            ckpt_path='./saved_ckpt/moffusion_H2.pth',

        ):
        self.model = 'moffusion_H2'
        self.df_cfg = './configs/moffusion-H2.yaml'
        self.ckpt = ckpt_path
        
    def name(self):
        return 'MOFFUSION-H2-Option'
    

class MOFFUSIONTopoOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='mof_250k', res=32):
        # dataset - snet
        self.dataroot = dataroot
        self.res = res
        self.dataset_mode = dataset_mode
        
    def init_model_args(
            self,
            ckpt_path='./saved_ckpt/moffusion_topo.pth',

        ):
        self.model = 'moffusion_topo'
        self.df_cfg = './configs/moffusion-topo.yaml'
        self.ckpt = ckpt_path
        
    def name(self):
        return 'MOFFUSION-Topo-Option'

class MOFFUSIONTextOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='mof_250k', res=32):
        # dataset - snet
        self.dataroot = dataroot
        self.res = res
        self.dataset_mode = dataset_mode
        
    def init_model_args(
            self,
            ckpt_path='./saved_ckpt/moffusion_text.pth',

        ):
        self.model = 'moffusion_text'
        self.df_cfg = './configs/moffusion-text.yaml'
        self.ckpt = ckpt_path
        
    def name(self):
        return 'MOFFUSION-Text-Option'
    
############ END: all Opt classes ############

# get partial shape from range
def get_partial_shape_MOF(shape, xyz_dict, z=None):
    """
        args:  
            shape: input sdf. (B, 3, H, W, D)
            xyz_dict: user-specified range.
                x: left to right
                y: bottom to top
                z: front to back
    """
    x = shape
    device = x.device
    (x_min, x_max) = xyz_dict['x']
    (y_min, y_max) = xyz_dict['y']
    (z_min, z_max) = xyz_dict['z']
    
    # clamp to [-1, 1]
    x_min, x_max = max(-1, x_min), min(1, x_max)
    y_min, y_max = max(-1, y_min), min(1, y_max)
    z_min, z_max = max(-1, z_min), min(1, z_max)

    B, _, H, W, D = x.shape # assume D = H = W

    x_st = int( (x_min - (-1))/2 * H )
    x_ed = int( (x_max - (-1))/2 * H )
    
    y_st = int( (y_min - (-1))/2 * W )
    y_ed = int( (y_max - (-1))/2 * W )
    
    z_st = int( (z_min - (-1))/2 * D )
    z_ed = int( (z_max - (-1))/2 * D )
    
    # print('x: ', xyz_dict['x'], x_st, x_ed)
    # print('y: ', xyz_dict['y'], y_st, y_ed)
    # print('z: ', xyz_dict['z'], z_st, z_ed)

    # where to keep    
    x_mask = torch.ones(B, 4, H, W, D).bool().to(device)
    x_mask[:, :, :x_st, :, :] = False
    x_mask[:, :, x_ed:, :, :] = False
    
    x_mask[:, :, :, :y_st, :] = False
    x_mask[:, :, :, y_ed:, :] = False
    
    x_mask[:, :, :, :, :z_st] = False
    x_mask[:, :, :, :, z_ed:] = False
        
    shape_part = x.clone()
    shape_missing = x.clone()
    shape_part[~x_mask] = 0.2 # T-SDF
    shape_missing[x_mask] = 0.2
    
    ret = {
        'shape_part': shape_part,
        'shape_missing': shape_missing,
        'shape_mask': x_mask,
    }
    
    if z is not None:
        B, _, zH, zW, zD = z.shape # assume D = H = W

        x_st = int( (x_min - (-1))/2 * zH )
        x_ed = int( (x_max - (-1))/2 * zH )
        
        y_st = int( (y_min - (-1))/2 * zW )
        y_ed = int( (y_max - (-1))/2 * zW )
        
        z_st = int( (z_min - (-1))/2 * zD )
        z_ed = int( (z_max - (-1))/2 * zD )
        
        # where to keep    
        z_mask = torch.ones(B, 3, zH, zW, zD).to(device)
        z_mask[:, :, :x_st, :, :] = 0.
        z_mask[:, :, x_ed:, :, :] = 0.
        
        z_mask[:, :, :, :y_st, :] = 0.
        z_mask[:, :, :, y_ed:, :] = 0.
    
        z_mask[:, :, :, :, :z_st] = 0.
        z_mask[:, :, :, :, z_ed:] = 0.
        
        ret['z_mask'] = z_mask

    return ret

