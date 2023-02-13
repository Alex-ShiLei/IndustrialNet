r""" Helper functions """
import random

import torch
import numpy as np
import cv2
clock=0
clock1=0
def saveImag(tenSorVal,name):
    img_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(0)  # [0.4485, 0.4196, 0.3810]
    img_std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(0)  # [0.2687, 0.2641, 0.2719]
    tenSorVal=tenSorVal.squeeze()
    tenSorVal = tenSorVal.permute(1, 2, 0)
    tenSorVal = tenSorVal.cpu()
    tenSorVal=(tenSorVal*img_std)+img_mean
    tenSorVal=tenSorVal*255
    tenSorVal=np.uint8(tenSorVal.numpy())
    tenSorVal=cv2.cvtColor(tenSorVal,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./map/'+str(name)+'.bmp',tenSorVal)

def saveMask(tenSorVal, name):
    tenSorVal = tenSorVal.squeeze()
    tenSorVal = tenSorVal*255
    tenSorVal = tenSorVal.cpu()
    tenSorVal = np.uint8(tenSorVal.numpy())
    cv2.imwrite('./map/' + str(name) + '.bmp', tenSorVal)
def saveColorMap(tenSorVal,name,size=[]):
    tenSorVal=tenSorVal.squeeze()
    if len(size)!=0:
        tenSorVal=tenSorVal.reshape(size)
    map=tenSorVal.cpu().numpy()
    minVal=map.min()
    maxVal=map.max()
    span=maxVal-minVal
    val=(map-minVal+1e-6)/(span+1e-6)
    #print('----',val.shape)
    val=np.uint8(val*255)
    #print(val.shape)
    img=cv2.applyColorMap(val,cv2.COLORMAP_JET)
    cv2.imwrite('./map/'+str(name)+'.bmp',img)
def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def cam(tenSorVal,name):
    global clock
    tenSorVal=tenSorVal.squeeze()
    map=tenSorVal.cpu().numpy()
    minVal=map.min()
    maxVal=map.max()
    span=maxVal-minVal
    val=(map-minVal+1e-6)/(span+1e-6)
    #print('----',val.shape)
    val=np.uint8(val*255)
    #print(val.shape)
    img=cv2.applyColorMap(val,cv2.COLORMAP_JET)
    cv2.imwrite('./map/'+str(clock//3)+'_'+str(name)+'.bmp',img)
    clock+=1
def cam1(tenSorVal,name):
    global clock1
    tenSorVal=tenSorVal.squeeze()
    map=tenSorVal.cpu().numpy()
    minVal=map.min()
    maxVal=map.max()
    span=maxVal-minVal
    val=(map-minVal+1e-6)/(span+1e-6)
    #print('----',val.shape)
    val=np.uint8(val*255)
    #print(val.shape)
    img=cv2.applyColorMap(val,cv2.COLORMAP_JET)
    cv2.imwrite('./cor/'+str(clock1//3)+'_'+str(name)+'.bmp',img)
    clock1+=1