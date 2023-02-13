r""" Provides functions that builds/manipulates correlation tensors """
import torch
import common.utils as util
class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids,name=''):
        eps = 1e-5
        diffs=[]
        corrs = []
        sups=[]
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape#b,c,h,w
            corrI=[]
            diffI=[]
            realSupI=[]
            for j in range(len(support_feat)):#b
                queryIJ=query_feat[j].flatten(start_dim=1)#c,hw
                queryIJ_Val=queryIJ.norm(dim=0, p=2, keepdim=True) + eps
                queryIJNorm=queryIJ/queryIJ_Val
                supIJ=support_feat[j]#c,hw
                supIJ_Val=supIJ.norm(dim=0, p=2, keepdim=True) + eps
                supIJNorm=supIJ/supIJ_Val
                corr=(queryIJNorm.T).matmul(supIJNorm)#hw,hw
                maxIndex=corr.argmax(dim=1)#hw
                new_query=supIJNorm[:,maxIndex]#c,n
                diff=((new_query-queryIJNorm)**2).mean(dim=0,keepdim=True)#1,hw
                maxval=diff.amax(dim=1,keepdim=True)+torch.zeros_like(diff)
                minVal=diff.amin(dim=1,keepdim=True)+torch.zeros_like(diff)
                diff=(diff-minVal)/(maxval-minVal)
                diff=diff.unsqueeze(0)#1,2,hw
                diffI.append(diff)

                corr=corr.mean(dim=1,keepdim=True)
                corr=(corr.permute(1,0)).unsqueeze(0)#1,1,hw
                corrI.append(corr)

                resupJ=supIJ.mean(dim=1,keepdim=True)
                resupJ=resupJ.unsqueeze(0).expand(-1,-1,queryIJ.shape[-1])#1,c,hw

                queryIJ=queryIJ.unsqueeze(0)#1,c,hw
                resupJ=torch.cat([queryIJ,resupJ],dim=1)#1,2c,hw
                realSupI.append(resupJ)#b,2c,hw
            diffI=torch.cat(diffI,dim=0)
            diffI = diffI.reshape((diffI.shape[0], diffI.shape[1], queryShape[-2], queryShape[-1]))  # b,1,h,w
            corrI=torch.cat(corrI,dim=0)#b,1,h,w
            corrI=corrI.reshape((corrI.shape[0],corrI.shape[1],queryShape[-2],queryShape[-1]))#b,1,h,w
            realSupI=torch.cat(realSupI,dim=0)#b,2c,h,w
            realSupI=realSupI.reshape((realSupI.shape[0],realSupI.shape[1],queryShape[-2],queryShape[-1]))
            diffs.append(diffI)#.unsqueeze(0))#b,1,h,w
            corrs.append(corrI)#.unsqueeze(0))#b,2,h,w
            sups.append(realSupI)#1,b,c,h,w

        diff_l4 = torch.cat(diffs[-stack_ids[0]:],dim=1).contiguous()#n,c,h,w
        diff_l3 = torch.cat(diffs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()
        diff_l2 = torch.cat(diffs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()

        corr_l4 = torch.cat(corrs[-stack_ids[0]:],dim=1).contiguous()#n,c,h,w
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()

        sup_l4=sups[-stack_ids[0]:]#n,b,2c,h,w
        sup_l3=sups[-stack_ids[1]:-stack_ids[0]]
        sup_l2=sups[-stack_ids[2]:-stack_ids[1]]
        return [diff_l4, diff_l3, diff_l2],[corr_l4, corr_l3, corr_l2],[sup_l4,sup_l3,sup_l2]
