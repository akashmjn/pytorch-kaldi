# import neural_networks
# import torch
# from neural_networks import LSTM_cudnn, MultiRNNLayerBridge
# options = {"hidden_size":5,"num_layers":3,"bias":"false","batch_first":"false",
#         "project_size":5, "layer_norm":"false","dropout":0.2,"bidirectional":"true"}
# lstm = LSTM_cudnn(options,2)
# x = torch.randn(10,3,2)

import configparser
import numpy as np
from dataloader import read_lab_fea_loader, PyTorchKaldiDataset, FoldedChunkBatchSampler, PaddedBatchSampler, PytorchKaldiDataLoader
from utils import dict_fea_lab_arch

cfg = configparser.ConfigParser()
cfg.read("./test/train_train_sp_comb_hires_ep000_ck02.cfg")

fea_dict, lab_dict, arch_dict = dict_fea_lab_arch(cfg)

dataset = PyTorchKaldiDataset(fea_dict,lab_dict)
#dataset_dict = read_lab_fea_loader(fea_dict,lab_dict)
#train_sampler = FoldedChunkBatchSampler(dataset,16,40)
#valid_sampler = FoldedChunkBatchSampler(dataset,16,40,False)
#eval_sampler = EvalPaddedBatchSampler(dataset,16)

dataloader_train = PytorchKaldiDataLoader(dataset,'train',16,40)
dataloader_valid = PytorchKaldiDataLoader(dataset,'valid',16,40)
dataloader_eval = PytorchKaldiDataLoader(dataset,'forward',16)
