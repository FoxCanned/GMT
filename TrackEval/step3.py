"""重新构建数据集格式
    """
import os
import shutil
import numpy as np
def mkdirc(path):
    if not os.path.exists(path):
        os.mkdir(path)

    
root = '/home/zyh/data/TrackEval/vision'
scenes = os.listdir(root)
for scene in scenes:
    txt_root = os.path.join(root,scene)
    gt_data = np.loadtxt(txt_root, delimiter=',', dtype=str)
    st = ''
    for i,line in enumerate(gt_data):
        fnum = int(line[0])
        x,y,z,w =  float(line[2]),float(line[3]),float(line[4]),float(line[5])
        id = int(float(line[1]))
        st += '{:d},{:.1f},{:.2f},{:.2f},{:.2f},{:.2f},1,-1,-1,-1\n'.format(fnum,id,
        x, y,z-x , w-y)
    with open(os.path.join(txt_root), 'w') as f:
        f.write(st)
        

