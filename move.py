import os
import shutil

i = 0
with open('data/datasets/train.txt', 'r', encoding='utf-8') as f:
    for ann in f.readlines():
        i = i + 1
        if i < 20 :
            continue
        if i > 100 :
            break
        ann = ann.strip('\n')       #去除文本中的换行符
        sourse = '/nfs/liujiaxuan/data/pascalvoc2012/VOCdevkit/VOC2012/JPEGImages/'+ann+'.jpg'
        shutil.copy(sourse,'data/image/')