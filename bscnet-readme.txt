BSCNet整体基于mmseg框架实现，请依据机台配置及mmseg官方文档进行环境搭建
https://github.com/open-mmlab/mmsegmentation


bscnet训练：
CUDA_VISIBLE_DEVICES=0,1 PORT=15200 ./tool/dist_train.py ./configs/bscnet/bscnet.py 2