#!/usr/bin/env python

import shutil
import os
import random
import glob
from tqdm import tqdm

train_percent = 0.85


labelfilepath = 'datasets/yolo/labelsTMP/'
imagefilepath = 'datasets/yolo/imagesTMP/'

trainimagepath = 'datasets/yolo/images/train'
valimagepath = 'datasets/yolo/images/val'
trainlabelpath = 'datasets/yolo/labels/train'
vallabelpath = 'datasets/yolo/labels/val'

total_file = glob.glob(os.path.join(imagefilepath, '*.jpg'))

num = len(total_file)
list = range(num)
tr = int(num*train_percent)
train = random.sample(list, tr)

txt_file_names = os.listdir(labelfilepath)
image_file_names = os.listdir(imagefilepath)

for i in tqdm(list):
    image_file_name = image_file_names[i]
    txt_file_name =  os.path.splitext(image_file_name)[0] + ".txt"
    if  i in train:
        shutil.move(os.path.join(labelfilepath, txt_file_name), trainlabelpath)
        shutil.move(os.path.join(imagefilepath, image_file_name), trainimagepath)
    else:
        shutil.move(os.path.join(labelfilepath, txt_file_name), vallabelpath)
        shutil.move(os.path.join(imagefilepath, image_file_name), valimagepath)
        
        
        
