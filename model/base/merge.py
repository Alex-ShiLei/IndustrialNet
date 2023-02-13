import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
class BasicConv(nn.Sequential):
    def __init__(self, inCh, outCh):
        super(BasicConv, self).__init__()
        self.conv0 = nn.Conv2d(inCh, outCh, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(outCh)
        self.relu0 = nn.ReLU()
class BasicConv1x1(nn.Sequential):
    def __init__(self, inCh, outCh):
        super(BasicConv1x1, self).__init__()
        self.conv0 = nn.Conv2d(inCh, outCh, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(outCh)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(outCh, outCh, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outCh)
        self.relu1 = nn.ReLU()
        #self.drop=nn.Dropout2d(0.3)

class ShotConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(ShotConv, self).__init__()
        self.conv0 =nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=1), nn.BatchNorm2d(outCh),nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=3, dilation=2, padding=2),nn.BatchNorm2d(outCh), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=3, dilation=4, padding=4),nn.BatchNorm2d(outCh),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(outCh*3, outCh, kernel_size=3,padding=1), nn.BatchNorm2d(outCh),nn.ReLU())
        self.drop=nn.Dropout2d(0.2)
    def forward(self,x):
        x0=self.conv0(x)
        x1=self.conv1(x)
        x2=self.conv2(x)
        x= self.conv3(torch.cat([x0,x1,x2],dim=1))
        return self.drop(x)
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)
class LayerConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(LayerConv, self).__init__()
        self.conv0 =nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=1), nn.BatchNorm2d(outCh),nn.ReLU())
        self.conv0_1 = nn.Sequential(nn.Conv2d(outCh*2, outCh, kernel_size=1), nn.BatchNorm2d(outCh),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=3,padding=1),nn.BatchNorm2d(outCh), nn.ReLU())
        self.conv1_1 = nn.Sequential(nn.Conv2d(outCh*2, outCh, kernel_size=1), nn.BatchNorm2d(outCh),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(outCh, outCh, kernel_size=3,padding=1), nn.BatchNorm2d(outCh),nn.ReLU())
    def forward(self,x):
        x0=self.conv0(x)
        x0mean=x0.mean(dim=(2,3),keepdim=True)+torch.zeros_like(x0)
        x0=self.conv0_1(torch.cat([x0,x0mean],dim=1))
        x1=self.conv1(x)
        x1mean=x1.mean(dim=(2,3),keepdim=True)+torch.zeros_like(x1)
        x1=self.conv1_1(torch.cat([x1,x1mean],dim=1))
        x= self.conv2(x0+x1)
        return x
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.adaptive_max_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # visMap.heatmap(scale, win='abc')
        # time.sleep(2)
        channel_att_sum = torch.sigmoid(channel_att_sum)
        scale = channel_att_sum.unsqueeze(2).unsqueeze(3)
        return x * scale


class MergeConv1(nn.Module):
    def __init__(self, inplan1, inplan2,inplan3, outplan):
        super(MergeConv1, self).__init__()
        self.conv1 = nn.Conv2d(inplan1, outplan, kernel_size=1)
        self.conv2 = nn.Conv2d(inplan2, outplan, kernel_size=1)
        self.conv3 = nn.Conv2d(inplan3, outplan, kernel_size=1)

        self.chATT = ChannelGate(outplan, 4)
        self.merge = BasicConv(outplan, outplan)
        self.merge2 = BasicConv(outplan, outplan)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x1, x2,x3):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        out = self.merge(x1+x2+x3)
        x = self.chATT(out)
        out = self.merge2(x)
        return self.drop(out)


class MergeConv(nn.Module):
    def __init__(self, inplan1, inplan2, outplan):
        super(MergeConv, self).__init__()
        self.conv1 = nn.Conv2d(inplan1, outplan, kernel_size=1)
        self.conv2 = nn.Conv2d(inplan2, outplan, kernel_size=1)
        self.merge = BasicConv(2 * outplan, outplan)
        self.merge2 = BasicConv(outplan, outplan)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x1, x2):
        # shape=x1.shape
        x1 = self.conv1(x1)
        # x2= F.interpolate(x2, size=(shape[2], shape[3]), mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        out = self.merge(torch.cat([x1, x2], dim=1))
        out = self.merge2(out)
        return self.drop(out)


class MergeDown(nn.Module):
    def __init__(self, inplan1, inplan2, outplan):
        super(MergeDown, self).__init__()
        self.conv1 = BasicConv(inplan1, outplan)
        self.conv2 = nn.Conv2d(inplan2, outplan, kernel_size=1)
        self.merge = BasicConv(2 * outplan, outplan)
        self.convout = BasicConv(outplan, outplan)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = self.merge(torch.cat([x1, x2], dim=1))#+x1
        val = self.convout(out)
        return val

class qsSim(nn.Module):
    def __init__(self, inplan):
        super(qsSim, self).__init__()
        self.conv2 = nn.Conv2d(inplan, 1, kernel_size=1,bias=False)
    def forward(self, x):
        x = self.conv2(x)
        val=torch.sigmoid(x)
        return val
class sqOut(nn.Module):
    def __init__(self, inCh, outCh):
        super(sqOut, self).__init__()
        self.conv0 =nn.Sequential(nn.Conv2d(inCh, outCh, kernel_size=1),nn.BatchNorm2d(outCh),nn.ReLU())
        self.conv1=BasicConv(outCh,outCh)
        self.conv2=BasicConv(outCh,outCh)
    def forward(self,x):
        x0=self.conv0(x)
        x1=self.conv1(x0)
        x=self.conv2(x0+x1)
        return  x

class merge(nn.Module):
    def __init__(self, shot, nfeatures=[2048 * 2, 1024 * 2, 512 * 2], nsimlairy=[3, 6, 4],criter=None):
        super(merge, self).__init__()
        self.shot = shot
        self.nsimlairy = nsimlairy
        self.query_sup_corrConv = []
        self.qs_layer_corrConv = []
        self.sim_layer_corrConv = []
        self.diff_layer_corrConv = []
        self.qs_sim = []
        self.criter=criter
        for num in nfeatures:
            self.query_sup_corrConv.append(qsSim(num))
        for num in nsimlairy:
            self.qs_layer_corrConv.append(BasicConv1x1(num, 256))
            self.sim_layer_corrConv.append(BasicConv1x1(num, 256))
            self.diff_layer_corrConv.append(BasicConv1x1(num, 256))
        self.query_sup_corrConv = nn.ModuleList(self.query_sup_corrConv)
        self.qs_layer_corrConv = nn.ModuleList(self.qs_layer_corrConv)
        self.sim_layer_corrConv = nn.ModuleList(self.sim_layer_corrConv)
        self.diff_layer_corrConv = nn.ModuleList(self.diff_layer_corrConv)
        self.sqShotConv4 = ShotConv(256, 256)
        self.sqShotConv3 = ShotConv(256, 256)
        self.sqShotConv2 = ShotConv(256, 256)

        self.diffShotConv4 = ShotConv(256, 256)
        self.diffShotConv3 = ShotConv(256, 256)
        self.diffShotConv2 = ShotConv(256, 256)

        self.simShotConv4 = ShotConv(256, 256)
        self.simShotConv3 = ShotConv(256, 256)
        self.simShotConv2 = ShotConv(256, 256)
        self.conv4 = MergeConv1(256,256, 256, 256)
        self.conv3 = MergeConv1(256, 256, 256, 256)
        self.conv2 = MergeConv1(256, 256, 256, 256)
        self.conv43 = MergeConv(256, 256, 512)
        self.conv432 = MergeDown(512, 256, 512)
        self.decoder1 = sqOut(512, 256)

        self.decoder4 = nn.Sequential(nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 2, (3, 3), padding=(1, 1)))

        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 2, (3, 3), padding=(1, 1)))

        self.decoder2 = nn.Sequential(nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 2, (3, 3), padding=(1, 1)))

        self.decoderOut = nn.Sequential(nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 2, (3, 3), padding=(1, 1)))

    def forward(self, qur_sups, sims,diffs,gt=None):
        qs_sim = []
        corr_sim = []
        diff_val = []
        for s in range(self.shot):
            sup = qur_sups[s]
            simS = sims[s]
            diffS=diffs[s]
            diffList=[]
            supsimList = []
            simList = []
            for l in range(len(sup)):  # L4,L3,L2
                supL = sup[l]
                supLsim = []
                for i in range(len(supL)):
                    supSim = self.query_sup_corrConv[l](supL[i])  # b,1,h,w
                    supLsim.append(supSim)
                supLsim = torch.cat(supLsim, dim=1)  # b,n,h,w
                supLsim = self.qs_layer_corrConv[l](supLsim)
                supsimList.append(supLsim)  # L,b,128,h,w
                #print('---------',simS[l].shape,diffS[l].shape)
                simL = self.sim_layer_corrConv[l](simS[l])
                simList.append(simL)  # L,b,128,h,w
                diffL = self.diff_layer_corrConv[l](diffS[l])
                diffList.append(diffL)
            diff_val.append(diffList)
            qs_sim.append(supsimList)  # s,l,b,128,h,w
            corr_sim.append(simList)

        diff_lyVal = [diff_val[0][i] for i in range(len(self.nsimlairy))]  # l,b,128,h,w
        corr_lySim = [corr_sim[0][i] for i in range(len(self.nsimlairy))]  # l,b,128,h,w
        qs_lySim = [qs_sim[0][i] for i in range(len(self.nsimlairy))]  # l,b,128,h,w
        for ly in range(len(self.nsimlairy)):
            for s in range(1,self.shot):
                corr_lySim[ly]=corr_lySim[ly]+(corr_sim[s][ly])
                qs_lySim[ly]=qs_lySim[ly]+(qs_sim[s][ly])
                diff_lyVal[ly]=diff_lyVal[ly]+(diff_val[s][ly])
        x4 = self.conv4(self.diffShotConv4(diff_lyVal[0]/self.shot),self.simShotConv4(corr_lySim[0]/self.shot),
                        self.sqShotConv4(qs_lySim[0]/self.shot))

        x3 = self.conv3(self.diffShotConv3(diff_lyVal[1]/self.shot),self.simShotConv3(corr_lySim[1]/self.shot),
                        self.sqShotConv3(qs_lySim[1]/self.shot))

        x2 = self.conv2(self.diffShotConv2(diff_lyVal[2]/self.shot),self.simShotConv2(corr_lySim[2]/self.shot),
                        self.sqShotConv2(qs_lySim[2]/self.shot))
        x4 = F.interpolate(x4, x3.size()[-2:], mode='bilinear', align_corners=True)
        x43 = self.conv43(x4, x3)
        x43 = F.interpolate(x43, x2.size()[-2:], mode='bilinear', align_corners=True)
        x432 = self.conv432(x43, x2)
        d1 = self.decoder1(x432)
        upsize = (d1.shape[-1] * 2,) * 2
        d1 = F.interpolate(d1, upsize, mode='bilinear', align_corners=True)
        d2 = self.decoderOut(d1)
        if self.training:
            lossSize=x432.size()[-2:]
            gt=gt.unsqueeze(1).float()
            gtOut = F.interpolate(gt, upsize, mode='nearest')
            gtOut=gtOut.squeeze(1)
            gtOut=gtOut.long()
            gt=F.interpolate(gt, lossSize, mode='nearest')
            gt=gt.squeeze(1)
            gt=gt.long()
            x2=self.decoder2(x2)
            x3 = self.decoder3(x3)
            x4 = self.decoder4(x4)
            x2=F.interpolate(x2, lossSize,mode='bilinear', align_corners=True)
            x3 = F.interpolate(x3, lossSize, mode='bilinear', align_corners=True)
            x4 = F.interpolate(x4, lossSize, mode='bilinear', align_corners=True)
            loss=0.3*(self.criter(x2,gt)+self.criter(x3,gt)+self.criter(x4,gt))+self.criter(d2,gtOut)
            return d2,loss
        else:
            gt = gt.unsqueeze(1).float()
            gtOut = F.interpolate(gt, upsize, mode='nearest')
            gtOut=gtOut.squeeze(1)
            gtOut = gtOut.long()
            loss=self.criter(d2, gtOut)
            return d2,loss
