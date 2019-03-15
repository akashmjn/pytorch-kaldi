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
from dataloader import read_lab_fea_loader, PytorchKaldiDataset, FoldedChunkBatchSampler, PaddedBatchSampler, PytorchKaldiDataLoader, SpeakerChunkSampler, MeetingChunkSampler
from neural_networks import AffineTransformDeep
from utils import dict_fea_lab_arch

def test_dataloader():
    cfg = configparser.ConfigParser()
    cfg.read("./test/forward_eval_hires_ep008_ck00.cfg")
    fea_dict, lab_dict, arch_dict = dict_fea_lab_arch(cfg)

    dataset = PytorchKaldiDataset(fea_dict,lab_dict)
    #dataset_dict = read_lab_fea_loader(fea_dict,lab_dict)
    #train_sampler = FoldedChunkBatchSampler(dataset,16,40)
    #valid_sampler = FoldedChunkBatchSampler(dataset,16,40,False)
    #eval_sampler = EvalPaddedBatchSampler(dataset,16)
    eval_spk_chunk_sampler = SpeakerChunkSampler(dataset,"./test/info.csv",16)
    eval_mtg_chunk_sampler = MeetingChunkSampler(dataset,"./test/info.csv",16)

    dataloader_train = PytorchKaldiDataLoader(dataset,'train',16,40)
    dataloader_valid = PytorchKaldiDataLoader(dataset,'valid',16,40)
    dataloader_eval = PytorchKaldiDataLoader(dataset,'forward',16)

    return dataloader_train, dataloader_valid, dataloader_eval, eval_spk_chunk_sampler, eval_mtg_chunk_sampler 

dl_train, dl_valid, dl_eval, eval_spk_chunk_sampler, eval_mtg_chunk_sampler = test_dataloader()
options = {'fea_dim':3,'act':'linear'}
A1 = AffineTransformDeep(options,5)
x = torch.randn(4,5)