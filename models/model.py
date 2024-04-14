from __future__ import division
from errno import ESTALE
import os
from pickle import TRUE 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
import math
from typing import DefaultDict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
# import torchvision.models as models
import numpy as np
from torch.nn.parameter import Parameter
import timm
from torch import Tensor

class pfcsa(nn.Module):
    def __init__(self):
        super(pfcsa, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmod=nn.Sigmoid()
        self.softmax_c=nn.Softmax(dim=1)
        self.softmax_s=nn.Softmax(dim=-1)
    def forward(self,x):
        n,b,h,w=x.shape
        x_c=self.gap(x)
        x_c=self.softmax_c(x_c)
        x_s=x.mean(axis=1).unsqueeze(1)
        x_s=self.softmax_s(x_s)
        return x*x_c*x_s+x

class resnet_base(nn.Module):
    def __init__(self,backbone,mode,att):
        super(resnet_base, self).__init__()
        if mode=='s':
            self.backbone=timm.create_model(backbone, pretrained=False,num_classes=0)
            self.backbone.conv1= nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3,bias=False)
        else:
            self.backbone=timm.create_model(backbone, pretrained=True,num_classes=0)
        if backbone in ['resnet18','resnet34']:
            self.dim=64
        elif backbone in ['resnet50','resnet101','resnet152']:
            self.dim=256
        if att:
            self.at1=pfcsa()
            self.at2=pfcsa()
            self.at3=pfcsa()
            self.at4=pfcsa()
        else:
            self.at1=nn.Identity()
            self.at2=nn.Identity()
            self.at3=nn.Identity()
            self.at4=nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,x):
        x=self.backbone.conv1(x)
        x=self.backbone.bn1(x)
        x=self.backbone.act1(x)
        x=self.backbone.maxpool(x)
        x=self.backbone.layer1(x)
        x=self.at1(x)
        x=self.backbone.layer2(x)
        x=self.at2(x)
        x=self.backbone.layer3(x)
        x=self.at3(x)
        x=self.backbone.layer4(x)
        x=self.at4(x)
        x=self.gap(x).squeeze(-1).squeeze(-1)
        return x

class Cos_at(nn.Module):
    #codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, q_dim,kv_dim,o_dim,dp):
        super(Cos_at, self).__init__()
        self.q_dim=q_dim
        self.kv_dim=kv_dim
        self.o_dim=o_dim
        self.q = nn.Linear(self.q_dim,o_dim)
        self.k = nn.Linear(self.kv_dim, o_dim)
        self.v = nn.Linear(self.kv_dim, o_dim)
        self.W = nn.Linear(o_dim, self.kv_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_q, x_kv):
        batch_size = x_q.size(0)
        channel_n=x_q.size(1)
        q = self.q(x_q).view(batch_size, self.o_dim, 1)
        k = self.k(x_kv).view(batch_size, 1,self.o_dim,)
        v = self.v(x_kv).view(batch_size, self.o_dim, 1)
     
        energy = torch.matmul(q, k)
        attention = self.softmax(energy)
        out = torch.matmul(attention,v).view(batch_size, self.o_dim)
        W_out = self.W(out)
        y = W_out+x_kv
        return torch.cat([y,x_q],1)
class resnet(nn.Module):
    def __init__(self,class_num,model,mode,att,fusion,f_dim,f_dp):
        super(resnet, self).__init__()
        self.model=model
        self.mode=mode 
        self.fusion=fusion            
        if model in ['resnet18','resnet34']:
            self.dim=64
        elif model in ['resnet50','resnet101','resnet152']:
            self.dim=256
        if len(self.mode)>2:
            self.s_model=resnet_base(self.model,'s',att)
            self.a_model=resnet_base(self.model,'a',att)
            self.g_model=resnet_base(self.model,'g',att)          
            if fusion:
                self.fusion_sa=Cos_at(self.dim*8,self.dim*8,f_dim,f_dp)
                self.fusion_og=Cos_at(self.dim*8*2,self.dim*8,f_dim,f_dp)
            self.fc_b = nn.Linear(self.dim*24,class_num)
        elif len(self.mode)==1:
            if "S" in self.mode:
                self.s_model=resnet_base(self.model,'s',att)
            else:
                self.a_or_g_model=resnet_base(self.model,'a',att)
            self.fc_b = nn.Linear(self.dim*8,class_num)   
        else:
            if "S" not in self.mode:
                self.a_model=resnet_base(self.model,'a',att)
                self.g_model=resnet_base(self.model,'g',att)
                if fusion:
                    self.fusion_ag=Cos_at(self.dim*8,self.dim*8,f_dim,f_dp)
            elif "A" not in self.mode:
                self.s_model=resnet_base(self.model,'s',att)
                self.g_model=resnet_base(self.model,'g',att)
                if fusion:
                    self.fusion_sg=Cos_at(self.dim*8,self.dim*8,f_dim,f_dp)
            else:
                self.s_model=resnet_base(self.model,'s',att)
                self.a_model=resnet_base(self.model,'a',att)
                if fusion:
                    self.fusion_sa=Cos_at(self.dim*8,self.dim*8,f_dim,f_dp)
            self.fc_b = nn.Linear(self.dim*16,class_num)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,s_sample,a_sample,g_sample):
        if len(self.mode)>2:
            s_x=self.s_model(s_sample)
            a_x=self.a_model(a_sample)
            g_x=self.g_model(g_sample)
            if self.fusion:
                sa_x=self.fusion_sa(a_x,s_x)
                sag_x=self.fusion_og(sa_x,g_x)
                x=sag_x
            else:
                x=torch.cat([s_x,a_x,g_x],1)
        elif len(self.mode)==1:
            if "S" in self.mode:
                x=self.s_model(s_sample)
            elif "A" in self.mode:
                x=self.a_or_g_model(a_sample)           
            else:              
                x=self.a_or_g_model(g_sample)
        else:
            if "S" not in self.mode:
                a_x=self.a_model(a_sample)
                g_x=self.g_model(g_sample)
                if self.fusion:
                    ag_x=self.fusion_ag(a_x,g_x)
                    x=ag_x
                else:
                    x=torch.cat([a_x,g_x],1)
            elif "A" not in self.mode:
                s_x=self.s_model(s_sample)
                g_x=self.g_model(g_sample)
                if self.fusion:
                    gs_x=self.fusion_sg(g_x,s_x)
                    x=gs_x
                else:
                    x=torch.cat([g_x,s_x],1)
            else:
                s_x=self.s_model(s_sample)
                a_x=self.a_model(a_sample)
                if self.fusion:
                    as_x=self.fusion_sa(a_x,s_x)
                    x=as_x
                else:
                    x=torch.cat([a_x,s_x],1)
        x=self.fc_b(x)
        return x
