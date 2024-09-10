import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamvidDataset(CustomDataset):

    CLASSES = ('sky', 'building', 'column pole','road', 'sidewalk', 'tree', 'signsymbol', 'fence', 'car','pedestrian', 'bicyclist','void')               
    PALETTE = [[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64,64,128], [64, 0, 128],
               [ 64, 64, 0], [0, 128, 192],[0,0,0]]  
               
    #CLASSES = ('void','sky', 'building', 'column pole','road', 'sidewalk', 'tree', 'signsymbol', 'fence', 'car','pedestrian', 'bicyclist')               
    #PALETTE = [[0,0,0],[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64,64,128], [64, 0, 128],
    #           [ 64, 64, 0], [0, 128, 192]]              
    def __init__(self,**kwargs):
        super(CamvidDataset, self).__init__(
            img_suffix='png',
            seg_map_suffix='png',
            ignore_index=11,
            **kwargs)
        assert osp.exists(self.img_dir)
        


#fast_seg:CAMVID_CLASSES = ['Sky','Building','Column-Pole',
#                  'Road','Sidewalk','Tree',
#                  'Sign-Symbol','Fence','Car',
#                  'Pedestrain','Bicyclist','Void']
#CAMVID_CLASS_COLORS = [(128, 128, 128),(128, 0, 0),(192, 192, 128),
#    (128, 64, 128),(0, 0, 192),(128, 128, 0),
#    (192, 128, 128),(64, 64, 128),(64, 0, 128),
#    (64, 64, 0),(0, 128, 192),(0, 0, 0),]