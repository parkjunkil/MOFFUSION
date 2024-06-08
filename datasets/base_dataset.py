
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets
import omegaconf
# from configs.paths import dataroot


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

# Create Dataset for MOF-SDF
def CreateDataset(opt):
    configs = omegaconf.OmegaConf.load(opt.vq_cfg)
    mparam = configs.model.params
    ddconfig = mparam.ddconfig

    from datasets.mof_dataset import MOFDataset
    train_dataset = MOFDataset()
    test_dataset = MOFDataset()
    train_dataset.initialize(opt, 'train',
                                res=opt.res,
                                dataset_name=opt.dataset_mode,
                                num_ch=ddconfig.in_channels)
    
    test_dataset.initialize(opt, 'test',
                            res=opt.res,
                            dataset_name=opt.dataset_mode,
                            num_ch=ddconfig.in_channels)
    
    opt.encoders = train_dataset.encoders



    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
