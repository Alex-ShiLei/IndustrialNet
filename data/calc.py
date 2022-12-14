import os
import numpy as np
import cv2

filepath = '/home/kexin/Project/DataSet/VOCdevkit/VOC2012/JPEGImages'  # 数据集目录
nameFile='/home/kexin/Project/sxw/hsnet-main/data/splits/pascal/trn/fold0.txt'
namesFile=open(nameFile)
pathDir=namesFile.readlines()
pathDir=[name.split('__')[0]+'.jpg' for name in pathDir]
numImg=len(pathDir)

R_channel = 0.0
G_channel = 0.0
B_channel = 0.0
for idx in range(numImg):
    filename = pathDir[idx]
    print('m',idx,filename)
    img = cv2.imread(os.path.join(filepath, filename))
    img=cv2.resize(img,(473,473))
    img=img/255.0
    R_channel = R_channel + np.sum(img[:, :, 2])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 0])

num = numImg * 473 * 473
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(numImg):
    filename = pathDir[idx]
    print('d', idx, filename)
    img = cv2.imread(os.path.join(filepath, filename))
    img=cv2.resize(img,(473,473))
    img=img/255.0
    R_channel = R_channel + np.sum((img[:, :, 2] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 0] - B_mean) ** 2)

R_var = R_channel / num
G_var = G_channel / num
B_var = B_channel / num
print("RGB_mean [%f,%f,%f]" % (R_mean, G_mean, B_mean))
print("RGB_std  [%f,%f,%f]" % (R_var**(0.5), G_var**(0.5), B_var**0.5))