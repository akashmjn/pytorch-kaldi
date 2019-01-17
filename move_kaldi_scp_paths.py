import os,sys
import re,glob 
import pdb

assert len(sys.argv)>=3, "Reqd args: curr_eg_path, new_eg_path, (optional: glob_string)"
# Remove trailing '/' if present
curr_eg_path, new_eg_path = sys.argv[1].rstrip('/'), sys.argv[2].rstrip('/')

if len(sys.argv)==4:
    glob_string = sys.argv[3]
else:
    glob_string = "./*/*.scp"

## Print info message 
print("\n ## Info: Only run in terminal egs directory, such as kaldi/egs/ami/s5b. Pass \"glob_string\" inside quotes ##\n")

## ASSUMES all paths are absolute paths
assert (curr_eg_path[0]=='/') and (new_eg_path[0]=='/'), "Please only use absolute paths!"

# Glob for all scp files 
gl = glob.glob(glob_string)

for scpfile in gl:

    print("Processing scp file: {}".format(scpfile))

    with open(scpfile,'r') as f:
        scplines = f.readlines()
    
    # Replace <curr_eg_path> with <new_eg_path>  
    scplines_sub = [ re.sub(curr_eg_path,new_eg_path,l) for l in scplines ] 
    
    with open(scpfile,'w') as f:
        f.writelines(scplines_sub)    


