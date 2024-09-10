_base_ = ['../_base_/default_runtime.py']
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='drbav2_3_0403_Backbone_ELPPM_BFMadd',
        num_classes=19, 
        planes=32, 
        augment=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=-1,
        channels=64,
        num_convs=1,
        kernel_size=3,
        concat_input=False,
        dropout_ratio=0,
        num_classes=19,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507])),
    auxiliary_head = [
        dict(
            type='FCNHead',
            in_channels=64,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                    class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                  1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                  1.0865, 1.0955, 1.0865, 1.1529, 1.0507])), 
        dict(
            type='FCNHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                    class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                  1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                  1.0865, 1.0955, 1.0865, 1.1529, 1.0507])),                                   
        dict(
            type='STDCHead',
            in_channels=2,
            channels=2,
            num_convs=0,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]), 
    ],
    
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True, paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='YOLOX', warmup='exp', by_epoch=False, warmup_by_epoch=False, warmup_ratio=1, warmup_iters=1000, num_last_epochs=3, min_lr_ratio=0.00001)#1e-5

log_config = dict( interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    
total_iters = 120000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')

# dataset settings
dataset_type = 'CityscapesDataset'
data_root='/data1/wlj2/mmseg_camvid/mmsegmentation/data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),  
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
            #dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    pin_momery=False,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))


custom_hooks = [dict(type='ExpMomentumEMAHook', resume_from=None, priority=49)] 

find_unused_parameters=True     
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()

work_dir = './point3_work_dirs/drbav2_3_0403_Backbone_ELPPM_BFMadd_bw10'