##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import sys
import configparser
import os
from utils import read_args_command_line,dict_fea_lab_arch,is_sequential_dict,compute_cw_max,model_init,optimizer_init,forward_model,progress
from data_io import load_counts,read_lab_fea
import numpy as np
import random
import torch
from distutils.util import strtobool
import time
from scipy.ndimage.interpolation import shift
import kaldi_io
from torch.nn.utils.rnn import pad_sequence
from dataloader import PytorchKaldiDataset, PytorchKaldiDataLoader


# Reading chunk-specific cfg file (first argument-mandatory file) 
cfg_file=sys.argv[1]

if not(os.path.exists(cfg_file)):
     sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
     sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)
    
  
# Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002)
[section_args,field_args,value_args]=read_args_command_line(sys.argv,config)
    
# list all the features, labels, and architecture actually used in the model section
[fea_dict,lab_dict,arch_dict]=dict_fea_lab_arch(config)

# check automatically if the model is sequential
seq_model=is_sequential_dict(config,arch_dict)

# Setting torch seed
seed=int(config['exp']['seed'])
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Reading config parameters
use_cuda=strtobool(config['exp']['use_cuda'])
save_gpumem=strtobool(config['exp']['save_gpumem'])
multi_gpu=strtobool(config['exp']['multi_gpu'])
is_production=strtobool(config['exp']['production'])

to_do=config['exp']['to_do']
info_file=config['exp']['out_info']

model=config['model']['model'].split('\n')

forward_outs=config['forward']['forward_out'].split(',')
forward_normalize_post=list(map(strtobool,config['forward']['normalize_posteriors'].split(',')))
forward_count_files=config['forward']['normalize_with_counts_from'].split(',')
require_decodings=list(map(strtobool,config['forward']['require_decoding'].split(',')))

start_time = time.time()

dataset = PytorchKaldiDataset(fea_dict,lab_dict)

if to_do=='train':
    max_seq_length=int(config['batches']['max_seq_length_train']) #*(int(info_file[-13:-10])+1) # increasing over the epochs
    batch_size=int(config['batches']['batch_size_train'])
    dataloader = PytorchKaldiDataLoader(dataset,to_do,batch_size,max_seq_length)

if to_do=='valid':
    max_seq_length=int(config['batches']['max_seq_length_valid'])
    batch_size=int(config['batches']['batch_size_valid'])
    dataloader = PytorchKaldiDataLoader(dataset,to_do,batch_size,max_seq_length)

if to_do=='forward':
    max_seq_length=-1 # do to break forward sentences
    batch_size=64
    dataloader = PytorchKaldiDataLoader(dataset,to_do,batch_size)

elapsed_time_reading=time.time() - start_time 

# converting numpy tensors into pytorch tensors and put them on GPUs if specified
start_time = time.time()

# if not(save_gpumem) and use_cuda:
#    data_set=torch.from_numpy(data_set).float().cuda()
# else:
#    data_set=torch.from_numpy(data_set).float() 
   
elapsed_time_load=time.time() - start_time 


# Reading model and initialize networks
inp_out_dict=fea_dict

[nns,costs]=model_init(inp_out_dict,model,config,arch_dict,use_cuda,multi_gpu,to_do)
   
# optimizers initialization
optimizers=optimizer_init(nns,config,arch_dict)

# pre-training
for net in nns.keys():
  pt_file_arch=config[arch_dict[net][0]]['arch_pretrain_file']
  
  if pt_file_arch!='none':        
      checkpoint_load = torch.load(pt_file_arch)
      nns[net].load_state_dict(checkpoint_load['model_par'])
      optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
      optimizers[net].param_groups[0]['lr']=float(config[arch_dict[net][0]]['arch_lr']) # loading lr of the cfg file for pt

if to_do=='forward':
    
    post_file={}
    for out_id in range(len(forward_outs)):
        if require_decodings[out_id]:
            out_file=info_file.replace('.info','_'+forward_outs[out_id]+'_to_decode.ark')
        else:
            out_file=info_file.replace('.info','_'+forward_outs[out_id]+'.ark')
        post_file[forward_outs[out_id]]=kaldi_io.open_or_fd(out_file,'wb')


start_time = time.time()

loss_sum=0
err_sum=0
N_batches = dataloader.batch_sampler.N_batches

for i, inp in enumerate(dataloader):   
    
    max_len = inp.shape[0]

    # use cuda
    if use_cuda:
        inp=inp.cuda()

    if to_do=='train':
        # Forward input, with autograd graph active
        outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)
        
        for opt in optimizers.keys():
            optimizers[opt].zero_grad()
        outs_dict['loss_final'].backward()
        # Gradient Clipping (th 0.5)
        grad_max_norm, grad_med_norm, grad_clip_norm = 0.0, np.inf, 0.5
        for net in nns.keys():
            grad_norms = torch.stack(
                        [p.grad.data.norm(2) for p in nns[net].parameters() if p.grad is not None]
                      )
            grad_max_norm = max(torch.max(grad_norms).item(),grad_max_norm)
            grad_med_norm = min(torch.median(grad_norms).item(),grad_med_norm)
            torch.nn.utils.clip_grad_norm_(nns[net].parameters(), grad_clip_norm)
        grad_max_norm, grad_med_norm = round(grad_max_norm,3), round(grad_med_norm,4)

        for opt in optimizers.keys():
            if not(strtobool(config[arch_dict[opt][0]]['arch_freeze'])):
                optimizers[opt].step()
    else:
        batch_size = inp.shape[1]
        with torch.no_grad(): # Forward input without autograd graph (save memory)
            outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)

    if to_do=='forward':
        if not seq_model: # TODO(akash) implement batched forward for non-seq models
            raise Exception("Batched forward not implemented for non-seq models yet")
        # get number of frames for each utt in batch
        utt_ids = dataloader.batch_sampler.utt_ids[i]
        nframes = [dataset.get_utt(uid).shape[0] for uid in utt_ids]
        
        # save only unpadded part for each utt in batch
        for out_id in range(len(forward_outs)):
            out_save=outs_dict[forward_outs[out_id]].data.cpu().numpy()
            out_dim=out_save.shape[-1]
            out_save=out_save.reshape(-1,batch_size,out_dim)
            for j, out_save_j_len in enumerate(nframes):
                # save kth utterance in batch - truncate padding to original len 
                out_save_j = out_save[:out_save_j_len,j,:]
                if forward_normalize_post[out_id]:
                    # read the config file
                    counts = load_counts(forward_count_files[out_id])
                    out_save_j=out_save_j-np.log(counts/np.sum(counts))             
                    
                # save the output    
                kaldi_io.write_mat(post_file[forward_outs[out_id]], out_save_j, utt_ids[j])
    else:
        # for printing instantaneous values
        batch_loss = round(outs_dict['loss_final'].item(),3)
        batch_err = round(outs_dict['err_final'].item(),3)
        loss_sum += batch_loss
        err_sum += batch_err
       
    # Progress bar
    if to_do == 'train':
        status_string="Training |L:{}, Err:{}, Gmax/med:{}/{}|Len:{}| (Batch {}/{})".format(batch_loss,batch_err,
                                                        grad_max_norm,grad_med_norm,max_len,i+1,N_batches)
    if to_do == 'valid':
      status_string="Validating |L:{}, Err:{}|Len:{}| (Batch {}/{})".format(batch_loss,batch_err,max_len,i+1,N_batches)
    if to_do == 'forward':
      status_string="Forwarding |Len:{}| (Batch {}/{})".format(max_len,i+1,N_batches)
    
    progress(i, N_batches, status=status_string)

elapsed_time_chunk=time.time() - start_time 

loss_tot=loss_sum/N_batches
err_tot=err_sum/N_batches

# clearing memory
del inp, outs_dict

# save the model
if to_do=='train':
     for net in nns.keys():
         checkpoint={}
         checkpoint['model_par']=nns[net].state_dict()
         checkpoint['optimizer_par']=optimizers[net].state_dict()
         out_file=info_file.replace('.info','_'+arch_dict[net][0]+'.pkl')
         torch.save(checkpoint, out_file)
     
if to_do=='forward':
    for out_name in forward_outs:
        post_file[out_name].close()
     
# Write info file
with open(info_file, "w") as text_file:
    text_file.write("[results]\n")
    if to_do!='forward':
        text_file.write("loss=%s\n" % loss_tot)
        text_file.write("err=%s\n" % err_tot)
    text_file.write("elapsed_time_read=%f (reading dataset)\n" % elapsed_time_reading)
    text_file.write("elapsed_time_load=%f (loading data on pytorch/gpu)\n" % elapsed_time_load)
    text_file.write("elapsed_time_chunk=%f (processing chunk)\n" % elapsed_time_chunk)
    text_file.write("elapsed_time=%f\n" % (elapsed_time_chunk+elapsed_time_load+elapsed_time_reading))
text_file.close()

