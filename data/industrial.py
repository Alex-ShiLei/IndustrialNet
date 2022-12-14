r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
#from pycocotools.coco import orgCOCO
import json
import time
import numpy as np
import itertools
from collections import defaultdict
import sys
import cv2
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DatasetIndustrial(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark ='industrial'# 'industrial'
        self.shot = shot
        self.base_path=datapath
        self.annotion_path=self.base_path+'/data.json'
        print('load annotions in:',self.annotion_path,self.split)
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.class_ids = self.build_class_ids()
        self.n_cls=len(self.class_ids)
        print(self,self.class_ids,self.n_cls)
        self.img_metadata_classwise = self.build_img_metadata_classwise(self.base_path)
        self.clsName=self.img_metadata_classwise['clsName']
        self.clsDic=self.img_metadata_classwise['clsDic']
        #self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return 3500 if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self,data_path):
        with open(data_path+'/data.json', 'r') as f:
            img_metadata_classwise =json.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self,mask_path):
        gt=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        gt[gt>0]=int(1)
        return torch.tensor(gt)

    def load_frame(self):
        cls_index=torch.randint(self.n_cls,(1,))
        class_sample =self.class_ids[cls_index[0].item()] #np.random.choice(self.class_ids, 1, replace=False)[0]
        cls_len=len(self.clsDic[self.clsName[class_sample]]['bad'])
        name_index=torch.randint(cls_len,(1,))
        query_name=self.clsDic[self.clsName[class_sample]]['bad'][name_index]
        #query_name = np.random.choice(self.clsDic[self.clsName[class_sample]]['bad'], 1, replace=False)[0]
        query_img = Image.open(self.base_path+'/'+query_name).convert('RGB')
        query_mask = self.read_mask(self.base_path+'/'+str.replace(query_name,'.png','.bmp'))

        org_qry_imsize = query_img.size
        sup_index=[]
        sup_len=len(self.clsDic[self.clsName[class_sample]]['good'])
        s_index=torch.randint(sup_len,(1,))
        sup_index.append(s_index[0].item())
        while len(sup_index)<self.shot:
            s_index = torch.randint(sup_len, (1,))[0]
            if s_index in sup_index:
                continue
            sup_index.append(s_index.item())
        support_paths =[self.clsDic[self.clsName[class_sample]]['good'][x] for x in sup_index]#np.random.choice(self.clsDic[self.clsName[class_sample]]['good'], self.shot, replace=False)
        support_names=[]
        support_imgs = []
        for support_name in support_paths:
            support_imgs.append(Image.open(self.base_path+'/'+support_name).convert('RGB'))
            support_names.append(os.path.basename(support_name))
        #print(name_index,query_name, '---------', sup_index, support_names)
        return query_img, query_mask, support_imgs, os.path.basename(query_name), support_names, class_sample, org_qry_imsize

