import numpy as np
from termcolor import colored, cprint

import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.base_dataset import BaseDataset

from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import joblib

import utils


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class MOFDataset(BaseDataset):

    def initialize(self, opt, phase='train', res=32, dataset_name='mof_250k', num_ch=4):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        self.num_ch = num_ch
        self.dataset_name = dataset_name

        # Assign paths for MOF data files
        dataroot = opt.dataroot
        SDF_dir = f'{dataroot}/250k/sdfs/resolution_{res}'
        lcd_path = f'{dataroot}/250k/lcd_data.txt'
        H2_path = f'{dataroot}/250k/H2_data.txt'
        sa_path = f'{dataroot}/250k/sa_data.txt'
        vf_path = f'{dataroot}/250k/vf_data.txt'
        text_path = f'{dataroot}/250k/text_data.txt'

        file_list = f'{dataroot}/250k/splits/{phase}_split_{dataset_name}.txt'

        # Prepare encoders for MOF_Constructor if not assigned
        sdf_files = os.listdir(SDF_dir)
        topo_list = []
        node_list = ['N0']
        edge_list = []

        for sdf in sdf_files:
            mof_name = sdf[:-4]
            topo, node1, node2, edge = split_mof(mof_name)
            if topo not in topo_list:
                topo_list.append(topo)
            if node1 not in node_list:
                node_list.append(node1)
            if node2 not in node_list:
                node_list.append(node2)
            if edge not in edge_list:
                edge_list.append(edge)

        if opt.encoders == None:

            enc_topo = LabelEncoder()
            enc_node = LabelEncoder()
            enc_edge = LabelEncoder()

            enc_topo.fit(topo_list)
            enc_node.fit(node_list)
            enc_edge.fit(edge_list)

            # save_encoders
            expr_dir = '%s/%s/' % (opt.logs_dir, opt.name)
            utils.util.mkdirs(expr_dir)
            joblib.dump(enc_topo, expr_dir+'topo_encoder.joblib',compress=9)
            joblib.dump(enc_node, expr_dir+'node_encoder.joblib',compress=9)
            joblib.dump(enc_edge, expr_dir+'edge_encoder.joblib',compress=9)

            encoders = OrderedDict()
            encoders['topo'] = enc_topo
            encoders['node'] = enc_node
            encoders['edge'] = enc_edge

            self.encoders = encoders
        else:
            self.encoders = opt.encoders

        # Load MOF SDF data
        self.model_list = []
        with open(file_list) as f:
            model_list_s = []
            for l in f.readlines():
                model_id = l.rstrip('\n')

                path = f'{SDF_dir}/{model_id}.npy'
                model_list_s.append(path)

            self.model_list += model_list_s

        # Load MOF property data (LCD, H2, SA, VF, Text)
        if lcd_path is not None:
            lcd_dict = {}
            with open(lcd_path, 'r') as f:
                for line in f.readlines():
                    mof, lcd = line.split()
                    lcd_dict[mof] = float(lcd)/100.0

            self.lcd_dict = lcd_dict

        if H2_path is not None:
            H2_dict = {}
            with open(H2_path, 'r') as f:
                for line in f.readlines():
                    mof, wc = line.split()
                    H2_dict[mof] = float(wc)/100.0
            self.H2_dict = H2_dict

        if sa_path is not None:
            sa_dict = {}
            with open(sa_path, 'r') as f:
                for line in f.readlines():
                    mof, sa = line.split()
                    sa_dict[mof] = float(sa)/10000.0
            self.sa_dict = sa_dict

        if vf_path is not None:
            vf_dict = {}
            with open(vf_path, 'r') as f:
                for line in f.readlines():
                    mof, vf = line.split()
                    vf_dict[mof] = float(vf)
            self.vf_dict = vf_dict     

        if text_path is not None:
            text_dict = {}
            with open(text_path, 'r') as f:
                for line in f.readlines():
                    mof, text = line.split(',')
                    text_dict[mof] = text
            self.text_dict = text_dict

        np.random.default_rng(seed=0).shuffle(self.model_list)

        self.model_list = self.model_list[:self.max_dataset_size]

        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')
        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        sdf_file = self.model_list[index]
        sdf = np.load(sdf_file).astype(np.float32)

        sdf = sdf[:self.num_ch]
        sdf = torch.Tensor(sdf).view(self.num_ch, self.res, self.res, self.res)

        # Normalize SDFs
        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf[0, :] = torch.clamp(sdf[0, :], min=-thres, max=thres)

        # Get MOF ids and corresponding properties
        mof_name = (sdf_file.split('/')[-1]).split('.')[0]
        mof_id = mof_to_id(self.encoders, mof_name)

        ret = {
            'sdf': sdf,
            'path': sdf_file,
            'id' : mof_id,
            'lcd' : self.lcd_dict[mof_name],
            'H2' : self.H2_dict[mof_name],
            'sa' : self.sa_dict[mof_name],
            'vf' : self.vf_dict[mof_name],
            'text' : self.text_dict[mof_name],
        }
        return ret

    def __len__(self):
        return self.N

    def name(self):
        return f'MOFDataset-{self.res}_{self.dataset_name}'

# Split MOF's name into each component (e.g., pcu+N16+E14 -> [pcu, N16, N0, E14])
def split_mof(mof_name):
    tokens = mof_name.split("+")

    if len(tokens)==3:
        topo  = tokens[0]
        node1 = tokens[1]
        node2 = 'N0'
        edge  = tokens[2]

    else:
        topo  = tokens[0]
        node1 = tokens[1]
        node2 = tokens[2]
        edge  = tokens[3]

    return topo, node1, node2, edge

# Convert MOF's name into each embeddings through encoders
def mof_to_id(encoders, mof_name):
    topo, node1, node2, edge = split_mof(mof_name)
    topo_id = encoders['topo'].transform([topo])
    node1_id = encoders['node'].transform([node1])
    node2_id = encoders['node'].transform([node2])
    edge_id = encoders['edge'].transform([edge])

    return np.hstack((topo_id, node1_id, node2_id, edge_id))

