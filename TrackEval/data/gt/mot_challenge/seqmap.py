import os
import shutil
import numpy as np
def mkdirc(path):
    if not os.path.exists(path):
        os.mkdir(path)
name = 'zMDMT-train'
root = './'+name
root_seq = './seqmaps'
scenes = os.listdir(root)
txt_root = os.path.join(root_seq,name+'.txt')
txt = 'name\n'
for scene in scenes:
    if '_' in scene:
        txt += scene+'\n'
with open(os.path.join(txt_root), 'w') as f:
    f.write(txt)
        
