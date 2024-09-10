import argparse
import os
from pathlib import Path
from functools import partial
import cv2
import numpy as np
import sys
import os.path as osp
import torch
import torch.nn.functional as F
import mmcv
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from mmdet.cv_core import (Config, load_checkpoint, FeatureMapVis, show_tensor, imdenormalize, show_img, imwrite,imshow,
                           traverse_file_paths)
from mmseg.models import build_segmentor
from mmseg.datasets.builder import build_dataset
from mmseg.datasets.pipelines import Compose
from mmseg.ops import resize
from mmengine.visualization import Visualizer


def BAFM_BRB(x_f):
    x_f[0]=x_f[0].unsqueeze(0)  
    x_f[1]=resize(input=x_f[1], size=x_f[0].shape[2:], mode='bilinear', align_corners=True)
    print(x_f[0].shape, x_f[1].shape)
    
    x_f1,x_f2=x_f[0].clone(),x_f[0].clone()
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
              x_f1[i,0,row,start:end+1]=x_f[0][i,0,row,start:end+1].mean()

        for col in range(w):
            index=(x_f[1][i,0,:,col]== 0).nonzero()
            if index==[]:
              continue
            for j in range(len(index)-1):
              start,end=index[j],index[j+1]
              x_f2[i,0,start:end+1,col]=x_f[0][i,0,start:end+1,col].mean()   
    
    return x_f1+x_f2
                
                
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='../demo', help='show img dir')
    parser.add_argument('--show', type=bool, default=False, help='show results')
    parser.add_argument(
        '--output_dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args
    
    
def forward(self, img, gt,img_metas=None,test_cfg=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)

    outs = self._decode_head_forward_test(x,img_metas)   
    output = F.softmax(outs, dim=1)
    output = resize( input=output, size=x[-3].shape[2:], mode='bilinear', align_corners=True)
    output = output.argmax(dim=1)
    output = output.cpu()
    
    gt_ = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
    gt_=resize(input=gt_, size=x[-3].shape[2:], mode='bilinear', align_corners=True)
    gt_ = gt_[0]
    
    
    
    outs_=[]
    masks_list=[8]
    mask_f=torch.zeros(gt_.shape)

    for mask in masks_list:
        mask_f += gt_==mask
    temp_feature_before=x[-5]*mask_f[0]
    temp_feature_after=x[-2]*mask_f[0]
    #temp_feature_after[:,:,output[0]==i]=temp_feature_after.max()  #####
    outs_.extend([temp_feature_before[0],temp_feature_after[0]])
    
    '''
    for i in masks_list:
        temp_feature_before=x[-5]*(gt_==i)[0]
        temp_feature_after=x[-2]*(gt_==i)[0]
        
        if abs((temp_feature_after-temp_feature_before).sum())<10:
          print("pass")
          print(abs((temp_feature_after-temp_feature_before).sum()))
          continue        
        outs_.extend([temp_feature_before[0],temp_feature_after[0]])
    '''
        
    return outs_
    


    ''' 
    boundary=self.auxiliary_head[0].forward_test(x,img_metas,test_cfg,**kwargs)
    boundary=resize(
        input=boundary,
        size=(360,720),
        mode='bilinear',
        align_corners=True)
    #print(img.shape,boundary[0].shape)

    seg_logits_vis=torch.sigmoid(boundary)[0].permute(1,2,0).cpu().detach().numpy()[:,:,1:2]
    '''
    #imshow(seg_logits_vis)

    return x[-3]
    
def create_model(cfg, use_gpu=True):
    print(cfg.model.keys())
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if use_gpu:
        model = model.cuda()
    model.forward = partial(forward, model)
    return model
    
def show_featuremap_from_datalayer(visualizer,model, is_show, output_dir):
    dataset = build_dataset(cfg.data.test)
    for item in dataset:
        gt=item['gt_semantic_seg']
        img_tensor = item['img']
        img_metas = item['img_metas'][0].data
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0) 
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)       


        boundary=model.auxiliary_head[2].get_boudary_targets_pyramid(torch.tensor(gt).unsqueeze(1))
        
        outs = model.forward(img_tensor[0].data.unsqueeze(0),gt[0]) 
        
        
        '''
        # show 
        y=BAFM_BRB([outs[-2],boundary]) 
        in_=outs[-2]
        outs=[]
        
        outs.append(in_)
        outs.append(y[0])  
        print(outs[0].shape,outs[1].shape)
        print(abs(outs[0]-outs[1]))
        print(y.shape)  
        '''

        for i in range(len(outs)): 
            #drawn_img = visualizer.draw_featmap(outs[i],img_orig, channel_reduction='select_max',resize_shape=[256,512],alpha=0.5)
            #drawn_img = visualizer.draw_featmap(outs[i],img_orig, channel_reduction='squeeze_mean',resize_shape=[256,512],alpha=0.5)
            drawn_img = visualizer.draw_featmap(outs[i],img_orig, channel_reduction=None,topk=3, resize_shape=[512,1024],alpha=0.5)
            if is_show:
                visualizer.show(drawn_img)
            print(filename.split("/")[-1]+"_%d"%i)
            visualizer.add_image(filename.split("/")[-1]+"_%d"%i, drawn_img)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = False
    is_show = args.show
    #init_shape = (2048,1024, 3)  
    init_shape = (960,720, 3)
    visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],save_dir=args.output_dir)
    
    #featurevis,model = create_featuremap_vis(cfg, use_gpu, init_shape)
    model=create_model(cfg,use_gpu)
    show_featuremap_from_datalayer(visualizer,model, is_show, args.output_dir)