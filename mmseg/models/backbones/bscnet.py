import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer, constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.utils import get_root_logger
from ..builder import BACKBONES

BatchNorm2d = nn.SyncBatchNorm
#BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

    
class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=planes, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, groups=planes, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn4 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual

        return self.relu(out)

class newelppm(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(newelppm, self).__init__()
        
        self.dim = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(outplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)
        self.pool8 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),)
        self.pool4 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),)
        self.pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),)                                      

        self.process2 = nn.Sequential(
                                    nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
                                    BatchNorm2d(outplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),)
        self.process4 = nn.Sequential(
                                    nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
                                    BatchNorm2d(outplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),)
        self.process8 = nn.Sequential(
                                    nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
                                    BatchNorm2d(outplanes, momentum=bn_mom),
                                    nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                    BatchNorm2d(outplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),)
                                    

    def forward(self, x):

        x = self.dim(x)
        pool_list = [self.pool2(x), self.pool4(x), self.pool8(x)]

        x_1 = F.interpolate(self.pool1(x), size=(pool_list[0].shape[-2], pool_list[0].shape[-1]), mode='bilinear')      
        x_2 = F.interpolate(self.process2(x_1 + pool_list[0]), size=(pool_list[1].shape[-2], pool_list[1].shape[-1]), mode='bilinear')        
        x_4 = F.interpolate(self.process4(x_2 + pool_list[1]), size=(pool_list[2].shape[-2], pool_list[2].shape[-1]), mode='bilinear')        
        x_8 = F.interpolate(self.process8(x_4 + pool_list[2]), size=(x.shape[-2], x.shape[-1]), mode='bilinear')
        
        x_f = x_8 + x
        
        return x_f 

class SeBFM2(nn.Module):
    def __init__(self, l_inplanes=64, h_inplanes=128, l_outplanes=64, h_outplanes=128):
        super(SeBFM2, self).__init__()
        
        self.l2h = nn.Sequential(nn.Conv2d(l_inplanes, h_outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                 BatchNorm2d(h_outplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)   
        self.h_seg = nn.Sequential(nn.Conv2d(h_outplanes, h_outplanes, kernel_size=3, groups=h_outplanes, padding=1, bias=False),
                                 BatchNorm2d(h_outplanes, momentum=bn_mom),
                                 nn.Conv2d(h_outplanes, h_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(h_outplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)
                                                                                                       
        self.h2l = nn.Sequential(nn.Conv2d(h_inplanes, l_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(l_outplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)            
        self.l_seg = nn.Sequential(nn.Conv2d(l_outplanes, l_outplanes, kernel_size=3, groups=l_outplanes, padding=1, bias=False),
                                 BatchNorm2d(l_outplanes, momentum=bn_mom),
                                 nn.Conv2d(l_outplanes, l_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(l_outplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)   
                                 
    def forward(self, x):
        #x[0]是低分辨率分支，x[1]是高分辨率分支
        batch_l, channel_l, height_l, width_l = x[0].size()
        batch_h, channel_h, height_h, width_h = x[1].size() 
               
        x_h2l = F.interpolate(self.h2l(x[1]), size=(height_l, width_l), mode='bilinear')
        l_out = x_h2l + x[0]
        l_out = self.l_seg(l_out)

        x_l2h = F.interpolate(self.l2h(x[0]), size=(height_h, width_h), mode='bilinear')           
        h_out = x_l2h + x[1]
        h_out = self.h_seg(h_out)
             
        return [l_out, h_out] 


class BAFM_CFF(nn.Module):
    def __init__(self):
        super(BAFM_CFF, self).__init__()
        
 
        self.conv1 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(128, momentum=bn_mom),
                                 nn.ReLU(inplace=True),) 

        self.dw3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1, bias=False),
                                 BatchNorm2d(128, momentum=bn_mom),
                                 nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(128, momentum=bn_mom),
                                 nn.ReLU(inplace=True),)                                 
    def forward(self, x):
        #x[0]是深层语义Fsd，x[1]是浅层语义Fss, x[2]是深层边界Fbd
        out = torch.cat((x[0], x[1]), dim=1)
        out = self.conv1(out)
        
        out_ = torch.nn.functional.adaptive_avg_pool2d(out, (1,1))
        out_ = torch.sigmoid(out_)
        out_ = out * out_
             
        return out_ 

class BAFM_BRB(nn.Module):
    def __init__(self):
        super(BAFM_BRB, self).__init__()
    
    def forward(self, x_f):
        # [x_f, x_bound_new]
        x_f[0]=F.interpolate(x_f[0],scale_factor=1/8)
        x_f[1]=F.interpolate(x_f[1],scale_factor=1/8)
        
        
        xh_lr,xh_rl=x_f[0].clone(),x_f[0].clone()
        xv_tb,xv_bt=x_f[0].clone(),x_f[0].clone()
        N,_,h,w=x_f[1].shape


        for i in range(N):
          for row in range(h):
            index=(x_f[1][i,0,row,:]== 0).nonzero()
            if index==[]:
              continue
            for j in range(len(index)-1):
              start, end=index[j], index[j+1]
              #x_f1[i,0,row,start:end+1]=x_f[i,0,row,start:end+1].mean()
              for temp in range(start+1, end):
                xh_lr[i, 0, row, temp] = x_f[0][i, 0, row, start:temp+1].mean()
                xh_rl[i, 0, row, temp] = x_f[0][i, 0, row, temp:end+1].mean()
            
        for i in range(N):
          for col in range(w):
            index=(x_f[1][i,0,:,col]== 0).nonzero()
            if index==[]:
              continue
            for j in range(len(index)-1):
              start,end=index[j],index[j+1]
              #x_f2[i,0,start:end+1,col]=x_f[i,0,start:end+1,col].mean()
              for temp in range(start+1, end):
                xv_tb[i, 0, temp, col] = x_f[0][i, 0, start:temp+1, col].mean()
                xv_bt[i, 0, temp, col] = x_f[0][i, 0, temp:end+1, col].mean()
        
                # sum
        return F.interpolate(xh_lr+xh_rl+xv_tb+xv_bt,scale_factor=8)
                
                
               
@BACKBONES.register_module()
class bscnet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2], num_classes=19, planes=32, head_planes=64, augment=False):
        super(bscnet, self).__init__()

        self.augment = augment
        self.norm_eval=False
        self.relu = nn.ReLU(inplace=True)

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),)
        
        self.layer1 = self._make_layer(block, planes, planes, layers[0], stride=2)
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)

        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 =  self._make_layer(block, planes * 8, planes * 16, 1, stride=2)        
      
        self.layer3_ = self._make_layer(block, planes * 2, planes * 2, 2)
        self.layer4_ = self._make_layer(block, planes * 2, planes * 2, 2)
        self.layer5_ = self._make_layer(block, planes * 2, planes * 4, 1)

        self.elppm = newelppm(planes * 16, planes * 4)
        
        self.SeBFM1 = SeBFM2(l_inplanes=128, h_inplanes=64, l_outplanes=128, h_outplanes=64)
        self.SeBFM2 = SeBFM2(l_inplanes=256, h_inplanes=64, l_outplanes=256, h_outplanes=64)

        self.boundary_head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=0)        
        
        self.BAFM_CFF = BAFM_CFF()
        self.BAFM_BRB = BAFM_BRB()       

                                    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        layers = []
        downsample = None
        if stride==2:
            downsample = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, groups=inplanes, bias=False,padding=1),
                                        BatchNorm2d(inplanes, momentum=bn_mom),
                                        nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom))
        elif inplanes != planes and stride == 1:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                       BatchNorm2d(planes, momentum=bn_mom))
        else:
            downsample = None
            
        layers.append(block(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        output = []

        x = self.conv1(x)
        x = self.layer1(x)
        x_s = self.layer2(x)

        output.append(x_s) #0
  
        xl = self.layer3(x_s)
        xh = self.layer3_(x_s)        
        #x[0]是低分辨率分支，x[1]是高分辨率分支
        xl, xh = self.SeBFM1([xl, xh]) 

        xl = self.layer4(xl)
        xh = self.layer4_(xh)
        xl, xh = self.SeBFM2([xl, xh]) 
                              
        output.append(xl)                 

        xl = self.layer5(xl)
        xl = F.interpolate(self.elppm(xl), size=[height_output, width_output], mode='bilinear')             
        xh = self.layer5_(xh)               
        
        x_backbone = xl + xh
        output.append(x_backbone)

        x_bound = F.interpolate(self.boundary_head(x_backbone), size=[height_output, width_output], mode='bilinear')
        output.append(x_bound)
        x_bound = x_bound[:,1:2,:,:]
        x_bound = torch.sigmoid(x_bound)
        x_bound_new = x_bound.clone()
        with torch.no_grad():
            x_bound_new[x_bound > 0.1] = 1
            x_bound_new[x_bound <= 0.1] = 0

        
        x_f = self.BAFM_CFF([x_backbone, x_s])
        x_out = self.BAFM_BRB([x_f, x_bound_new])
        
        output.append(x_out+x_backbone)
        
        return output

    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.ReLU):
                    constant_init(m, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(bscnet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    