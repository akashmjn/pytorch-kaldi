import os,sys

assert len(sys.argv)==3, "Reqd args: data_dir, data_xvec_dir. Eg: <python ..> ./data/train_sp ./data/train_sp_xvec"
data_dir, xvec_dir = sys.argv[1], sys.argv[2]

utt2spk = { l.split()[0]:l.split()[1] for l in open(os.path.join(data_dir,"utt2spk")).read().splitlines() }
spk2spk_xvector_scp = { l.split()[0]:l.split()[1] for l in open(os.path.join(xvec_dir,"spk_xvector.scp")).read().splitlines() }

utt_xvec_scp = open(os.path.join(xvec_dir,"xvector.scp")).read().splitlines()

utt2spk_xvector_scp = [ l.split()[0]+" "+spk2spk_xvector_scp[utt2spk[l.split()[0]]] for l in utt_xvec_scp ]

out_file = os.path.join(xvec_dir,"utt2spk_xvector.scp")
with open(out_file,'w') as f:
    f.write("\n".join(utt2spk_xvector_scp)+"\n")
print("Done! Written: {}".format(out_file))
