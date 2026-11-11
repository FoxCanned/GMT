import configparser
import os
import numpy as np

# name=MOTS20-09
# imDir=img1
# frameRate=30
# seqLength=525
# imWidth=1920
# imHeight=1080
# imExt=.jpg


name = 'vision-train'
root = './'+name
scenes = os.listdir(root)
for scene in scenes:
    if not '_' in scene:
        continue
    txt_root = os.path.join(root,scene,'gt','gt.txt')
    gt_data = np.loadtxt(txt_root, delimiter=',', dtype=str)
    fnum = int(gt_data[len(gt_data)-1][0])

    config = configparser.ConfigParser()
    #创建一个节点名称 DEFAULT; 然后写入字典中的对应的键值对
    config["Sequence"] = {'imDir':'img1',
                        'frameRate':'30',
                        'imWidth':'1920',
                        'imHeight':'1080',
                        'imExt':'.jpg'
                        }
    config["Sequence"] ["seqLength"] = str(fnum)
    config["Sequence"] ["name"] = scene
    os.remove(os.path.join(root,scene,'gt','seqinfo.ini'))
    with open(os.path.join(root,scene,'seqinfo.ini'), 'w') as configfile:
        config.write(configfile)                                    