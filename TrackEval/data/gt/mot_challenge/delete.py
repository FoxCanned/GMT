"""修改数据集中标准的错误
"""
import  numpy as np
import os
root = './vision-train/00051football_View2/gt/gt.txt'
gt_data = np.loadtxt(root, delimiter=',', dtype=str)
st = ''
for i,line in enumerate(gt_data):
    fnum = int(line[0])
    x,y,z,w =  int(line[2]),int(line[3]),int(line[4]),int(line[5])
    id = int(line[1])
    if id==70:
        continue
    st += '{:d},{:d},{:d},{:d},{:d},{:d},1,1,1\n'.format(fnum,id,
    x, y,z , w)
with open(root, 'w') as f:
    f.write(st)