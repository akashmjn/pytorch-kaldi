##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import kaldi_io
import numpy as np
import sys
from scipy.ndimage.interpolation import shift
import time


def load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length,fea_vec):
    if fea_vec:
        fea= { k:m for k,m in kaldi_io.read_vec_flt_ark('ark:copy-vector scp:'+fea_scp+' ark:- |'+fea_opts) }
    else:
        fea= { k:m for k,m in kaldi_io.read_mat_ark('ark:copy-feats scp:'+fea_scp+' ark:- |'+fea_opts) }

    lab= { k:v for k,v in kaldi_io.read_vec_int_ark('gunzip -c '+lab_folder+'/ali*.gz | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|')  if k in fea} # Note that I'm copying only the aligments of the loaded fea

    if fea_vec:
        fea={k: np.tile(v,[lab[k].shape[0],1]) for k, v in fea.items() if k in lab} # tile vector for all frames if alignment exists 
    else:
        fea={k: v for k, v in fea.items() if k in lab} # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")

    end_snt=0
    end_index=[]
    snt_name=[]
    fea_conc=[]
    lab_conc=[]

    # TODO: Fix memory inefficiency in using lists + concatentation here
    
    tmp=0
    utt_ids = list(fea.keys())
    np.random.shuffle(utt_ids)
    for k in utt_ids:

        #####
        # If the sequence length is above the threshold, we split it with a minimal length max/4
        # If max length = 500, then the split will start at 500 + (500/4) = 625. 
        # A seq of length 625 will be splitted in one of 500 and one of 125

        
        #if(len(fea[k]) > max_sequence_length) and max_sequence_length>0:
        if False: # TODO:(akash) clean up all these hacks 

          fea_chunked = []
          lab_chunked = []

          for i in range((len(fea[k]) + max_sequence_length - 1) // max_sequence_length):
            if(len(fea[k][i * max_sequence_length:]) > max_sequence_length + (max_sequence_length/4)):
              fea_chunked.append(fea[k][i * max_sequence_length:(i + 1) * max_sequence_length])
              lab_chunked.append(lab[k][i * max_sequence_length:(i + 1) * max_sequence_length])
            else:
              fea_chunked.append(fea[k][i * max_sequence_length:])
              lab_chunked.append(lab[k][i * max_sequence_length:])
              break

          for j in range(0, len(fea_chunked)):
            fea_conc.append(fea_chunked[j])
            lab_conc.append(lab_chunked[j])
            snt_name.append(k+'_split'+str(j))
            
        else:
          fea_conc.append(fea[k])
          lab_conc.append(lab[k])
          snt_name.append(k)

        tmp+=1
    
    # fea_zipped = zip(fea_conc,lab_conc)
    # fea_sorted = sorted(fea_zipped, key=lambda x: x[0].shape[0])
    # fea_conc,lab_conc = zip(*fea_sorted)
      
    for entry in fea_conc:
      end_snt=end_snt+entry.shape[0]
      end_index.append(end_snt)

    fea_conc=np.concatenate(fea_conc)
    lab_conc=np.concatenate(lab_conc)

    return [snt_name,fea_conc,lab_conc,np.asarray(end_index)] 


def context_window_old(fea,left,right):
 
 N_row=fea.shape[0]
 N_fea=fea.shape[1]
 frames = np.empty((N_row-left-right, N_fea*(left+right+1)))
 
 for frame_index in range(left,N_row-right):
  right_context=fea[frame_index+1:frame_index+right+1].flatten() # right context
  left_context=fea[frame_index-left:frame_index].flatten() # left context
  current_frame=np.concatenate([left_context,fea[frame_index],right_context])
  frames[frame_index-left]=current_frame

 return frames

def context_window(fea,left,right):
 
    N_elem=fea.shape[0]
    N_fea=fea.shape[1]
    
    fea_conc=np.empty([N_elem,N_fea*(left+right+1)])
    
    index_fea=0
    for lag in range(-left,right+1):
        fea_conc[:,index_fea:index_fea+fea.shape[1]]=np.roll(fea,lag,axis=0)
        index_fea=index_fea+fea.shape[1]
        
    fea_conc=fea_conc[left:fea_conc.shape[0]-right]
    
    return fea_conc


def load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,max_sequence_length,fea_vec=False):
  
  # open the file
  [data_name,data_set,data_lab,end_index]=load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length,fea_vec)

  # Context window
  if left!=0 or right!=0:
      data_set=context_window(data_set,left,right)

  end_index=end_index-left
  end_index[-1]=end_index[-1]-right

  # mean and variance normalization
  data_set_means, data_set_std = np.mean(data_set,axis=0), np.std(data_set,axis=0)
  data_set -= data_set_means # inplace operations for memory efficiency
  data_set /= data_set_std

  # Label processing
  data_lab=data_lab-data_lab.min()
  if right>0:
   data_lab=data_lab[left:-right]
  else:
   data_lab=data_lab[left:]   
  
  data_set=np.column_stack((data_set, data_lab))

   
  return [data_name,data_set,end_index]

def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts 



def read_lab_fea(fea_dict,lab_dict,cw_left_max,cw_right_max,max_seq_length):
    
    # Estimate total number of frames in this chunk by relatively quickly reading alignments
    fea_scp = list(fea_dict.values())[0][1]
    lab_folder, lab_opts = list(lab_dict.values())[0][1:3]
    with open(fea_scp) as f:
        ck_utt_ids = set([ s.split()[0] for s in f.readlines() ])
    ck_lab_nframes = [ v.shape[0] for k,v in kaldi_io.read_vec_int_ark(
            "gunzip -c {0}/ali*.gz | {1} {0}/final.mdl ark:- ark:-|".format(lab_folder,lab_opts)
                ) if k in ck_utt_ids ] # list of num frames for utterance in current chunk     
    # pre-allocate an np array to avoid memory wastage with concatentation
    ck_nframes, ck_nlabs = sum(ck_lab_nframes), len(lab_dict) 
    ck_fea_dim = sum([ int(v[-1]) for v in list(fea_dict.values())] )
    data_set = np.empty((ck_nframes,ck_fea_dim+ck_nlabs))
    labs = data_set[:,-ck_nlabs:]

    # TODO: leave some buffer in data_set matrix (for now ck_nframes is exact)
    # TODO: this works for single label, no cw case, untested for others 

    # seed ensures same relative utt order between load_chunk calls
    # absolute order will differ next time read_lab_fea called on this chunk
    ck_random_seed = np.random.randint(256)
 
    fea_index, data_name, data_end_index = 0, None, None
    for cnt_fea, fea in enumerate(fea_dict.keys()):
        
        # reading the features
        fea_scp, fea_opts = fea_dict[fea][1], fea_dict[fea][2]
        cw_left, cw_right = int(fea_dict[fea][3]), int(fea_dict[fea][4])
        fea_vec='vec' in fea
        fea_dim=int(fea_dict[fea].pop()) # hacked in fea_dim. bad practice to modify arg like this
        
        for cnt_lab, lab in enumerate(lab_dict.keys()):
            
            lab_folder, lab_opts = lab_dict[lab][1], lab_dict[lab][2]
            np.random.seed(ck_random_seed)
            # copy features into corresponding cols of data_set matrix
            [ data_name_fea,
              data_set[:,fea_index:fea_index+fea_dim+1],
              data_end_index_fea ]=load_chunk(fea_scp,fea_opts,lab_folder,
                                    lab_opts,cw_left,cw_right,max_seq_length,fea_vec)

            # move labels into end column - offset
            labs[:,cnt_lab] = data_set[:,fea_index+fea_dim]
            lab_dict[lab].append(ck_fea_dim+cnt_lab)

            # Checks if lab_names are the same for all the features
            if data_name is not None and not(data_name==data_name_fea):
                sys.stderr.write('ERROR: different sentence ids are detected for the different features. Plase check again input feature lists"\n')
                sys.exit(0)
            
            # Checks if end indexes are the same for all the features
            if data_end_index is not None and not(data_end_index==data_end_index_fea).all():
                sys.stderr.write('ERROR end_index must be the same for all the sentences"\n')
                sys.exit(0)
            
            data_name, data_end_index = data_name_fea, data_end_index_fea
    
        # update the offset values and fea_dict (used downstream by model_init)
        fea_dict[fea].extend([fea_index,fea_index+fea_dim,fea_dim])
        fea_index += fea_dim

    return [data_name,data_set,data_end_index]
# TODO: figure out the cw related manipulations later
"""
        if cw_left!=0 or cw_right!=0: 
            # making the same dimenion for all the features (compensating for different context windows)
            labs_fea=data_set_fea[cw_left_max-cw_left:data_set_fea.shape[0]-(cw_right_max-cw_right),-1]
            data_set_fea=data_set_fea[cw_left_max-cw_left:data_set_fea.shape[0]-(cw_right_max-cw_right),0:-1]
            data_end_index_fea=data_end_index_fea-(cw_left_max-cw_left)
            data_end_index_fea[-1]=data_end_index_fea[-1]-(cw_right_max-cw_right)
        else:
            labs_fea = data_set_fea[:,-1] # avoid memory intensive operations on data_set
            data_set_fea = data_set_fea[:,0:-1]

        if cnt_fea==0 and cnt_lab==0:
            data_set=data_set_fea
            labs=labs_fea
            data_end_index=data_end_index_fea
            data_end_index=data_end_index_fea
            data_name=data_name_fea
            
            fea_dict[fea].append(fea_index)
            fea_index=fea_index+data_set_fea.shape[1]
            fea_dict[fea].append(fea_index)
            fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
            
        else:
            if cnt_fea==0:
                labs=np.column_stack((labs,labs_fea))
            
            if cnt_lab==0:
                data_set=np.column_stack((data_set,data_set_fea)) # memory intensive step 
                fea_dict[fea].append(fea_index)
                fea_index=fea_index+data_set_fea.shape[1]
                fea_dict[fea].append(fea_index)
                fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
    
cnt_lab=0    
for lab in lab_dict.keys():
    lab_dict[lab].append(data_set.shape[1]+cnt_lab)
    cnt_lab=cnt_lab+1
        
data_set=np.column_stack((data_set,labs))
"""


