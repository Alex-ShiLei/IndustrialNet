r""" Industrial Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from .base.merge import merge
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.new_corelation_norm import Correlation

class IndustrialNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize,shot=1):
        super(IndustrialNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.shot=shot
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.cross_entropy_loss =nn.CrossEntropyLoss(weight=torch.tensor([0.2,0.8]).cuda())
        self.merge=merge(shot,nsimlairy=list(reversed(nbottlenecks[-3:])),criter=self.cross_entropy_loss)
    def forward(self, query_img, support_img,gt=None,name=''):
        sup_feats=[]#shot
        corrs=[]
        diffs=[]
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            for i in range(self.shot):
                support_feats = self.extract_feats(support_img[:,i,:,:,:], self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats = self.mask_feature(support_feats)
                diff,corr,sups = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids,name)#[corr_l4, corr_l3, corr_l2]
                diffs.append(diff)
                corrs.append(corr)#s,l,n,b,1,h,w
                sup_feats.append(sups)#s,l,n,b,2*c,h,w

        logit_mask,loss = self.merge(sup_feats,corrs,diffs,gt)
        #time.sleep(100)
        #print('____________________')
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[-2:], mode='bilinear', align_corners=True)

        return logit_mask,loss

    def mask_feature(self, features):#bchw
        bs=features[0].shape[0]
        for idx, feature in enumerate(features):
            feat=[]
            for i in range(bs):
                featI=feature[i].flatten(start_dim=1)#c,hw
                feat.append(featI)#[b,]ch,w
            features[idx] = feat#nfeatures ,bs,ch,w
        return features

    def predict_mask_nshot(self, batch, nshot):
        imgName=batch['query_name'][0].split('.')[0]
        #saveImag(batch['query_img'],imgName+'_query')
        #saveImag(batch['support_imgs'],imgName+'_sup')
        logit_mask,loss = self(batch['query_img'], batch['support_imgs'],batch['query_mask'],imgName)
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

        logit_mask_agg = logit_mask.argmax(dim=1)
        return logit_mask_agg

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
