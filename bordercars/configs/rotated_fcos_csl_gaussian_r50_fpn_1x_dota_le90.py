angle_version = 'le90'
checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_file='data/split_1024_dota1_0/test/images/',
        img_prefix='data/split_1024_dota1_0/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1024,
                    1024,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='DOTADataset',
        version='le90'),
    train=dict(
        ann_file='data/split_1024_dota1_0/trainval/annfiles/',
        img_prefix='data/split_1024_dota1_0/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(img_scale=(
                1024,
                1024,
            ), type='RResize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                flip_ratio=[
                    0.25,
                    0.25,
                    0.25,
                ],
                type='RRandomFlip',
                version='le90'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
            ], type='Collect'),
        ],
        type='DOTADataset',
        version='le90'),
    val=dict(
        ann_file='data/split_1024_dota1_0/trainval/annfiles/',
        img_prefix='data/split_1024_dota1_0/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1024,
                    1024,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='DOTADataset',
        version='le90'),
    workers_per_gpu=2)
data_root = 'data/split_1024_dota1_0/'
dataset_type = 'DOTADataset'
dist_params = dict(backend='nccl')
evaluation = dict(interval=1, metric='mAP')
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet',
        zero_init_residual=False),
    bbox_head=dict(
        angle_coder=dict(
            angle_version='le90',
            omega=1,
            radius=1,
            type='CSLCoder',
            window='gaussian'),
        bbox_coder=dict(angle_version='le90', type='DistanceAnglePointCoder'),
        center_sample_radius=1.5,
        center_sampling=True,
        centerness_on_reg=True,
        feat_channels=256,
        h_bbox_coder=dict(type='DistancePointBBoxCoder'),
        in_channels=256,
        loss_angle=dict(
            alpha=0.25, gamma=2.0, loss_weight=0.2, type='SmoothFocalLoss'),
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes=15,
        scale_angle=False,
        separate_angle=True,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='CSLRFCOSHead'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_thr=0.1),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=None,
    type='RotatedFCOS')
mp_start_method = 'fork'
opencv_num_threads = 0
optimizer = dict(lr=0.0025, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
resume_from = None
runner = dict(max_epochs=12, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            1024,
            1024,
        ),
        transforms=[
            dict(type='RResize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(img_scale=(
        1024,
        1024,
    ), type='RResize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        flip_ratio=[
            0.25,
            0.25,
            0.25,
        ],
        type='RRandomFlip',
        version='le90'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
