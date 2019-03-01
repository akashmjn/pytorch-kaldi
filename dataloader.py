import torch
import torch.nn as nn
import numpy as np
import os,sys
import time, warnings
import kaldi_io, pdb

from torch.utils.data import Dataset, DataLoader, sampler
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from utils import compute_cw_max
from data_io import read_lab_fea


def kaldi_reader(load_path,opts,mode):
    """
    Returns: 
        data_array: concatenated data from kaldi files 
        names: keys read from kaldi files
        names2idx: ordered map from names -> (idx1,idx2) range in data_array
    """
    if mode=='mat':
        data_gen = kaldi_io.read_mat_ark('ark:copy-feats scp:'+load_path+' ark:- |'+opts)
    elif mode=='vec':
        data_gen = ( (k,v.reshape(1,-1)) for k,v in 
                kaldi_io.read_vec_flt_ark('ark:copy-vector scp:'+load_path+' ark:- |'+opts) )
    elif mode=='lab':
        data_gen = ( (k,v.reshape(-1,1)) for k,v in 
        kaldi_io.read_vec_int_ark('gunzip -c '+load_path+'/ali*.gz | '+opts+' '+load_path+'/final.mdl ark:- ark:-|') )
    conc_list, names, names2idx = [], [], OrderedDict()
    prev_idx = 0
    for name, data in data_gen:
        names.append(name)
        conc_list.append(data)
        curr_idx = prev_idx + data.shape[0]
        names2idx[name] = (prev_idx,curr_idx)
        prev_idx = curr_idx
    
    data_array = np.concatenate(conc_list)
    
    return data_array, names, names2idx


def read_lab_fea_loader(fea_dict,lab_dict):
    """
    Returns: 
        dataset_dict{
            features: {f1:(data_array,names,names2idx) ...},
            labels: {l1:(data_array,names,names2idx) ...}
        }
    """

    dataset_dict = {'features':{},'labels':{}}

    for fea in fea_dict.keys():
        # Data for this chunk: array, names, names2idx map
        fea_name,load_path,opts,cw_left,cw_right,dim = fea_dict[fea]  
        if int(cw_left) > 0 or int(cw_right) > 0: # TODO: low priority 
            raise NotImplementedError("Context windows not supported yet") 
        mode = 'vec' if 'vec' in fea_name else 'mat' # TODO: Hack. Needs new fea_dict field 
        data_array, names, names2idx = kaldi_reader(load_path,opts,mode)
        if mode != 'vec':
            # Normalization
            means, std = np.mean(data_array,axis=0), np.std(data_array,axis=0)
            data_array -= means # inplace operations for memory efficiency
            data_array /= (std+1e-6)
        dataset_dict['features'][fea_name] = (data_array,names,names2idx)
        sys.stdout.write(
            "LOG (dataloader.py:read_lab_fea_loader) {} loaded: {} entries into {} array\n".format(
                fea_name,len(names),data_array.shape)
            )
    for lab in lab_dict.keys():
        # Labels for entire dataset: array, names, names2idx map
        lab_name,load_path,opts = lab_dict[lab]  
        data_array, names, names2idx = kaldi_reader(load_path,opts,'lab')
        # Label processing
        data_array -= data_array.min()
        dataset_dict['labels'][lab_name] = (data_array,names,names2idx)
        sys.stdout.write(
            "LOG (dataloader.py:read_lab_fea_loader) {} loaded: {} entries into {} array\n".format(
                lab_name,len(names),data_array.shape)
            )

    return dataset_dict    


class PytorchKaldiDataset(Dataset):
    """
    """

    def __init__(self, fea_dict, lab_dict):
        # read large matrix / bunch of matrices into memory
        dataset_dict = read_lab_fea_loader(fea_dict,lab_dict)
        # each value of dict is key: (array, names, names2idx)
        self.features_dict = dataset_dict['features']
        self.labels_dict = dataset_dict['labels']
        self._sanity_check_data()
        self.set_index()
        self._set_dims(fea_dict,lab_dict)
    
    def _set_dims(self,fea_dict,lab_dict):
        self.dims, dim_idx = [], 0
        for colname in self._index_colnames:
            dim = self._get_colname_data(colname)[0].shape[-1]
            self.dims.append(dim)
            # update appropriate fields of fea and lab dict based on fields in the batch
            # TODO: fix this at some point, model_init and forward_model have a dependency
            if colname in fea_dict and len(fea_dict[colname])==6: #TODO:HACK!
                fea_dict[colname].insert(-1,dim_idx) # adding range (dim_idx,dim_idx+dim)
                fea_dict[colname].insert(-1,dim_idx+dim)
                fea_dict[colname][-1] = int(fea_dict[colname][-1])
            elif colname in lab_dict and len(lab_dict[colname])==3:
                lab_dict[colname].append(dim_idx)
            dim_idx += dim
        self.batch_dim = sum(self.dims)

    def _get_colname_data(self,colname):
        return self.features_dict[colname] if colname in self.features_dict else self.labels_dict[colname]

    def _sanity_check_data(self):
        # TODO: Do some sanity checks b/w loaded features, labels
        ## Check if feature data names match
        def clean_warning_show(message,category,filename,lineno,line=None):
            return "WARNING {}:{} {}\n".format(os.path.basename(filename),lineno,message)
        warnings.formatwarning = clean_warning_show
        for fea, (_,fea_names,fea_names2idx) in self.features_dict.items():
            for lab, (_,lab_names,lab_names2idx) in self.labels_dict.items():
                ## Check if all feature data has corresponding labels
                missing_names = set(fea_names) - set(lab_names)
                if len(missing_names) > 0:
                    warnings.warn("Feature:{} has {} names not present in label:{}".format(
                        fea,len(missing_names),lab))
                    if 'spk' not in fea: # TODO:HACK
                        for missing_name in missing_names:
                            fea_names.remove(missing_name)
                            del fea_names2idx[missing_name]
                        warnings.warn("Feature:{} Removed {} missing names: {}".format(fea,len(missing_names),missing_names))
    
    def set_index(self,shuffle=False):
        # use one fea as reference to create frame-aligned table of indexes, names
        fea_reference = [fea for fea in self.features_dict.keys() if 'vec' not in fea][0] #TODO:HACK
        _, utt_ids, fea_reference_names2idx = self.features_dict[fea_reference]
        self._index_colnames = list(self.features_dict.keys())+list(self.labels_dict.keys())

        # dataset can be shuffled by shuffling names, names2idx 
        if shuffle: np.random.shuffle(utt_ids)
         
        # fill index arrays with corresponding idx,name from dataset dicts 
        index_list, index_names_list, nframes_list = [], [], [0]
        for utt_id in utt_ids:
            nframes = len(range(*fea_reference_names2idx[utt_id]))
            index_arr = np.empty((nframes,len(self._index_colnames)),dtype=int)
            index_names_arr = np.empty((nframes,len(self._index_colnames)),dtype=object)
            for j,colname in enumerate(self._index_colnames):
                _, _, names2idx = self._get_colname_data(colname)
                # TODO: if some mapping for names provided use that, else
                index_names_arr[:,j] = utt_id
                index_arr[:,j] = range(*names2idx[utt_id])
            index_list.append(index_arr)
            index_names_list.append(index_names_arr)
            nframes_list.append(nframes)

        utt_index_ranges = np.cumsum(nframes_list)
        self.utt2index = OrderedDict([ 
            (uid,(utt_index_ranges[i],utt_index_ranges[i+1])) for i,uid in enumerate(utt_ids)
            ])
        self.index = np.concatenate(index_list)
        self.index_names = np.concatenate(index_names_list)
    
    def __len__(self):
        return self.index.shape[0] # returns total number of frames present in chunk

    def __getitem__(self, idx):
        ## allow index to be N x batch size (for reading chunks)
        result = []
        for j,colname in enumerate(self._index_colnames):
            array, _, _ = self._get_colname_data(colname) 
            array_idx = self.index[:,j]
            array_idx = array_idx[idx]
            result.append(array[array_idx])
        return np.concatenate(result,axis=-1)
    
    def get_utt(self,utt_id):
        return self[range(*self.utt2index[utt_id])]


class FoldedChunkBatchSampler(sampler.Sampler):
    # Determines order in which loader reads samples from dataset 
    # Modify fea_dict w/ appropriate fea_indexes (probably do this in loader instead)
    def __init__(self,dataset,batch_size,max_seq_length,shuffle=True):
        self.dataset = dataset
        self.dataset.set_index(shuffle=True) # Randomize order of utterances
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.batch_dim = self.dataset.batch_dim
        # create array of indexes folded over by batch size  
        nframes = len(self.dataset) - (len(self.dataset)%batch_size)
        self.batch_indices = np.array(range(nframes))
        self.batch_indices = self.batch_indices.reshape(self.batch_size,-1)\
                                 .transpose((1,0))
        self.N_batches = int(self.batch_indices.shape[0]/self.max_seq_length)#ignores last batch 

    def __iter__(self):
        # As a hack, returning list w/ one element as batching is already included in indices
        for i in range(self.N_batches):
            yield [self.batch_indices[i*self.max_seq_length:(i+1)*self.max_seq_length,:]]
    
    def __len__(self):
        return self.N_batches

class PaddedBatchSampler(sampler.Sampler):
    # iterate through utterances and create padded batches
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_dim = self.dataset.batch_dim

        N_utts = len(self.dataset.utt2index)
        # last batch may have < batch_size utterances
        self.N_batches = int(N_utts/self.batch_size) \
            if N_utts%self.batch_size==0 else int(N_utts/self.batch_size)+1
        utt_ids, _ = zip(*sorted(
            list(dataset.utt2index.items()),key=lambda x: x[1][1]-x[1][0],reverse=True
            )) # sorting dataset.utt2index in decreasing length for efficient padding 
        self.utt_ids = [ utt_ids[i*batch_size:(i+1)*batch_size] for i in range(self.N_batches) ]

    def __iter__(self):
        for i in range(self.N_batches):
            yield [ np.array(range(*self.dataset.utt2index[uid]))
                      for uid in self.utt_ids[i] 
                   ] 
    
    def __len__(self):
        return self.N_batches


class PytorchKaldiDataLoader(DataLoader):

    def __init__(self,dataset,mode,batch_size,max_seq_length=None,num_workers=0):

        self.dataset = dataset 
        assert mode in ["train","valid","forward"], "Invalid mode passed!"
        self.mode = mode
        if mode=='train':
            self.batch_sampler = FoldedChunkBatchSampler(self.dataset,batch_size,max_seq_length)
        elif mode=='valid':
            self.batch_sampler = FoldedChunkBatchSampler(self.dataset,batch_size,max_seq_length,
                           shuffle=False)
        elif mode=='forward':
            self.batch_sampler = PaddedBatchSampler(self.dataset,batch_size)

        super(PytorchKaldiDataLoader, self).__init__(
            dataset=self.dataset,batch_sampler=self.batch_sampler,collate_fn=self.collate_fn,
            num_workers=num_workers,pin_memory=True
            )
    
    def collate_fn(self,batch_list):
        # returns tensor compatible w/ forward_model 
        if self.mode=='forward':
            batch = pad_sequence(
                [ torch.from_numpy(t).float() for t in batch_list ]
            ).contiguous()
        else:
            batch = torch.from_numpy(batch_list[0]).float()
        return batch

"""
Dataloader -> iterator of batches
    - Dataset
    - Sampler 
    - collate_fn

torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, 
    num_workers=0, collate_fn=<function default_collate>, pin_memory=False, 
    drop_last=False, timeout=0, worker_init_fn=None
    )

<tensor>inp = next(Dataloader) # Should work w/ forward_model [:,f1 f2 lab]

outs_dict = forward_model(
    fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,
    max_len,batch_size,to_do,forward_outs
    )

------- Rough code --------------

        # Randomize if the model is not sequential
        if to_do!='forward':
            if not(seq_model):
                np.random.shuffle(data_set)
            else:
                # fold long series of utterances into batch_size num of sequences 
                nframes_rounded, ndim = data_set.shape[0] - (data_set.shape[0]%batch_size), data_set.shape      [1]
                data_set = data_set[:nframes_rounded,:]
                data_set = data_set.reshape((batch_size,-1,ndim)).transpose((1,0,2))
                print("Reshaped chunk of {} frames to {}".format(nframes_rounded,str(data_set.shape)))
        if seq_model and to_do!='forward': inp_dim=data_set.shape[2]
        else: inp_dim=data_set.shape[1] # TODO: doesn't pass labels to dnn. see forward_model
        
        # ***** Minibatch Processing loop********
        N_snt=len(data_name)
        if to_do=='forward':
            N_batches=int(N_snt/batch_size) if N_snt%batch_size==0 else int(N_snt/batch_size)+1
            snt_lookup=[] # indexing into data_set via [ (snt_name,(start_idx,end_idx)) ]
            for i in range(N_snt): 
                if i==0: snt_lookup.append((data_name[i],(0,data_end_index[i]))) 
                else: snt_lookup.append((data_name[i],(data_end_index[i-1],data_end_index[i])))
            snt_lookup.sort(key=lambda x: x[1][1]-x[1][0],reverse=True) # sort in decreasing length
        elif seq_model:
            N_batches=int(data_set.shape[0]/max_seq_length) # TODO: ignores the last batch that's <         max_seq_length
        else:
            N_batches=int(data_set.shape[0]/batch_size)

        if seq_model:
            if to_do=='forward':
                batch_size = min(batch_size,N_snt-beg_batch) # for remainder if N_snt%batch_size!=0 
                # create batch of sentences from lookup (snt_name,(start_idx,end_idx)) 
                inp = pad_sequence(
                    [ data_set[name_idx_tup[1][0]:name_idx_tup[1][1],:] \
                        for name_idx_tup in snt_lookup[beg_batch:beg_batch+batch_size] ]
                ).contiguous()
                max_len = inp.shape[0]
            else:
                max_len = max_seq_length
                inp = data_set[beg_snt:beg_snt+max_len,:,:].contiguous()
                beg_snt += max_len
        else:
            # features and labels for batch i
            if to_do!='forward': # TODO(akash) support batched forward for non-seq models
                inp= data_set[beg_batch:end_batch,:].contiguous()
            else:
                snt_len=data_end_index[snt_index]-beg_snt
                inp= data_set[beg_snt:beg_snt+snt_len,:].contiguous()
                beg_snt=data_end_index[snt_index]
                snt_index=snt_index+1

        # update it to the next batch 
        beg_batch=end_batch
        end_batch=beg_batch+batch_size
------- Rough code --------------
"""
