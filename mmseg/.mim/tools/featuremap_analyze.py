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
def show_result(img,
                result,
                palette=None,
                win_name='',
                show=True,
                wait_time=0,
                out_file=None,
                opacity=1):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        #img = mmcv.imread(img)
        img = img.copy()
        seg = result

        palette = np.array(palette)


        #assert palette.shape[1] == 3
        #assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
            
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='../demo', help='show img dir')
    # 显示预测结果
    parser.add_argument('--show', type=bool, default=False, help='show results')
    # 可视化图片保存路径
    parser.add_argument(
        '--output_dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def forward(self, img, img_metas=None,test_cfg=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self._decode_head_forward_test(x,img_metas)
    output = F.softmax(outs, dim=1)
    output = resize(
        input=output,
        size=img.shape[2:],
        mode='bilinear',
        align_corners=True)
    output = output.argmax(dim=1)
    output = output.cpu().numpy()
    boundary=self.auxiliary_head[0].forward_test(x,img_metas,test_cfg,**kwargs)
    boundary=resize(
        input=boundary,
        size=(360,480),
        mode='bilinear',
        align_corners=True)
    #print(img.shape,boundary[0].shape)
    seg_logits_vis=torch.sigmoid(boundary)[0].permute(1,2,0).cpu().detach().numpy()[:,:,1:2]
    #imshow(seg_logits_vis)

    return output,seg_logits_vis

def forward_aux(self, img,img_metas=None,test_cfg=None,**kwargs):
    x = self.extract_feat(img)
    outs = self.auxiliary_head[0].forward_test(x,img_metas,test_cfg,**kwargs)
    return outs

def create_model(cfg, use_gpu=True):
    print(cfg.model.keys())
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


def create_featuremap_vis(cfg, use_gpu=True, init_shape=(320, 320, 3)):
    model = create_model(cfg, use_gpu)
    model.forward = partial(forward, model)
    featurevis = FeatureMapVis(model, use_gpu)
    featurevis.set_hook_style(init_shape[2], init_shape[:2])
    return featurevis,model


def _show_save_data(featurevis, img, img_orig, feature_indexs, filepath, is_show, output_dir):
    show_datas = []
    for feature_index in feature_indexs:
        feature_map = featurevis.run(img.copy(), feature_index=feature_index)[0]
        data = show_tensor(feature_map[0], resize_hw=img.shape[:2], show_split=False, is_show=False)[0]
        print(data.shape)
        print(feature_map.shape)
        feature_map=feature_map[0][1:2,:,:]
        print(feature_map.shape)
        feature_map[torch.sigmoid(feature_map)>=0.1]=1
        feature_map[torch.sigmoid(feature_map)<0.1]=0
        feature_map=feature_map.cpu().detach().numpy().transpose(1, 2, 0)
        imshow(feature_map)
        feature_map=feature_map*255.0
        cv2.imwrite('test.jpg', feature_map.astype(np.uint8))

        #am_data = cv2.addWeighted(data, 0.5, img_orig, 0.5, 0)
        show_datas.append(data)
    if is_show:
        show_img(show_datas)
    if output_dir is not None:
        filename = os.path.join(output_dir,
                                Path(filepath).name
                                )
        if len(show_datas) == 1:
            imwrite(show_datas[0], filename)
        else:
            for i in range(len(show_datas)):
                fname, suffix = os.path.splitext(filename)
                imwrite(show_datas[i], fname + '_{}'.format(str(i)) + suffix)


def show_featuremap_from_imgs(featurevis, feature_indexs, img_dir, mean, std, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    img_paths = traverse_file_paths(img_dir, 'jpg')


    for path in img_paths:
        data = dict(img_info=dict(filename=path), img_prefix=None)
        test_pipeline = Compose(cfg.data.test.pipeline)
        item = test_pipeline(data)
        img_tensor = item['img']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # 依然是归一化后的图片
        img_orig = imdenormalize(img, np.array(mean), np.array(std)).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, path, is_show, output_dir)


def show_featuremap_from_datalayer(featurevis, model,feature_indexs, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    dataset = build_dataset(cfg.data.test)


    for item in dataset:
        img_tensor = item['img']
        img_metas = item['img_metas'][0].data
        print(item)

        gt=item['gt_semantic_seg']
        print(gt[0].shape,gt[0].min(),gt[0].max())
        print(gt[0].shape)
        '''
        #boundart_gt=model.auxiliary_head[0].get_boundary(torch.tensor(gt).unsqueeze(1))
        #boundart_gt=model.auxiliary_head[0].get_boundary(torch.tensor(gt[0]).unsqueeze(0))
        
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0) 
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)              
        #boundart_gt = boundart_gt[0].cpu().numpy().transpose(1, 2, 0)*255.0 
        outs,boundary=model.forward(img_tensor[0].data.unsqueeze(0))
        show_result(img_orig,gt[0],palette=dataset.PALETTE )         
        show_result(img_orig,outs[0],palette=dataset.PALETTE ) 
        #filename1 = os.path.join(output_dir,img_metas['ori_filename'])                                   
        #imshow(boundart_gt.astype(np.uint8))
        #cv2.imwrite(filename1, boundart_gt.astype(np.uint8))
        #print(boundart_gt)
        #imshow(boundary)
        #cv2.imwrite(filename1, (boundary*255.0).astype(np.uint8))
        #_show_save_data(featurevis, img, img_orig, feature_indexs, filename, is_show, output_dir)
        '''
        img_tensor = item['img']
        img_metas = item['img_metas'][0].data
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, filename, is_show, output_dir)



if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = False
    is_show = args.show
    init_shape = (2048,1024, 3)  # 值不重要，只要前向一遍网络时候不报错即可
    feature_index = [208]  # 想看的特征图层索引(yolov3  218 214 210)

    featurevis,model = create_featuremap_vis(cfg, use_gpu, init_shape)

    show_featuremap_from_datalayer(featurevis,model, feature_index, is_show, args.output_dir)

    #mean = cfg.img_norm_cfg['mean']
    #std = cfg.img_norm_cfg['std']
    #show_featuremap_from_imgs(featurevis, feature_index, args.img_dir, mean, std, is_show, args.output_dir)
