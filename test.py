# import neural_networks
# import torch
# from neural_networks import LSTM_cudnn, MultiRNNLayerBridge
# options = {"hidden_size":5,"num_layers":3,"bias":"false","batch_first":"false",
#         "project_size":5, "layer_norm":"false","dropout":0.2,"bidirectional":"true"}
# lstm = LSTM_cudnn(options,2)
# x = torch.randn(10,3,2)

import configparser
import numpy as np
import torch
from dataloader import read_lab_fea_loader, PytorchKaldiDataset, FoldedChunkBatchSampler, PaddedBatchSampler, PytorchKaldiDataLoader
from neural_networks import AffineTransformDeep, LSTM, LSTM_membiased 
from utils import dict_fea_lab_arch

def test_dataloader():
    cfg = configparser.ConfigParser()
    cfg.read("./test/train_train_sp_comb_hires_ep000_ck02.cfg")
    fea_dict, lab_dict, arch_dict = dict_fea_lab_arch(cfg)

    dataset = PytorchKaldiDataset(fea_dict,lab_dict)
    #dataset_dict = read_lab_fea_loader(fea_dict,lab_dict)
    #train_sampler = FoldedChunkBatchSampler(dataset,16,40)
    #valid_sampler = FoldedChunkBatchSampler(dataset,16,40,False)
    #eval_sampler = EvalPaddedBatchSampler(dataset,16)

    dataloader_train = PytorchKaldiDataLoader(dataset,'train',16,40)
    dataloader_valid = PytorchKaldiDataLoader(dataset,'valid',16,40)
    dataloader_eval = PytorchKaldiDataLoader(dataset,'forward',16)

    return dataloader_train, dataloader_valid, dataloader_eval

dl_train, dl_valid, dl_eval = test_dataloader()
options = {'fea_dim':3,'act':'linear'}
A1 = AffineTransformDeep(options,5)
x = torch.randn(4,5)

options = {
    'lstm_lay':"512,512,512",
    'lstm_drop':"0.2,0.2,0.2",
    'lstm_use_batchnorm':"False,False,False",
    'lstm_use_laynorm':"False,False,False",
    'lstm_use_batchnorm_inp':"False",
    'lstm_use_laynorm_inp':"False",
    'lstm_act':"tanh,tanh,tanh",
    'lstm_orthinit':"True",
    'lstm_bidir':"True",
    'use_cuda':"False",
    'to_do':"train"
    }

L1 = LSTM(options,40)
L2 = LSTM_membiased(options,40,512)

inp = next(iter(dl_train))
s = inp[:,:,:512]
x = inp[:,:,512:552]
