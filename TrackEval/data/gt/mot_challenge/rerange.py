import os
import shutil
import numpy as np
def mkdirc(path):
    if not os.path.exists(path):
        os.mkdir(path)

root = './vision-train'
scenes = os.listdir(root)
for scene in scenes:
    if '_' in scene:
        continue
    txt_root = os.path.join(root,scene,'gt')
    txts = os.listdir(txt_root)
    for txt in txts:
        t_root = os.path.join(txt_root,txt)
        gt_data = np.loadtxt(t_root, delimiter=' ', dtype=str)
        st = ''
        for i,line in enumerate(gt_data):
            fnum = int(line[0])
            x,y,z,w =  int(line[2]),int(line[3]),int(line[4]),int(line[5])
            id = int(line[1])
            st += '{:d},{:d},{:d},{:d},{:d},{:d},1,1,1\n'.format(fnum,id,
            x, y,z , w)
        root1 = os.path.join(root,scene+"_"+txt.split('.')[0])
        root2 = os.path.join(root1,'gt')
        mkdirc(root1)
        mkdirc(root2)
        with open(os.path.join(root2,'gt.txt'), 'w') as f:
            f.write(st)
        
