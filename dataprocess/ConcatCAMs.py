#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import glob

names = glob.glob(r'C:\data\mineral\trainingrecords\CAMS\CAMS_on_test_mobilenet\*.png')
dst_path = r'C:\data\mineral\trainingrecords\CAMS\CAMS_cat'
os.makedirs(dst_path, exist_ok=True)
for name in names:
    this_mob = cv2.imread(name)
    res = (cv2.imread(name.replace('mobilenet', 'resnet')))[:, 256:]
    swin = (cv2.imread(name.replace('mobilenet', 'swin')))[:, 256:]
    show_img = np.hstack([this_mob, res, swin])
    this_name = name.split('\\')[-1]
    cv2.imwrite(dst_path + '\\' + this_name, show_img)


