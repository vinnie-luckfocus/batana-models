# Self-contained RTMPose-m config for 19-keypoint baseball fine-tuning.
# Adapted from mmpose rtmpose-m_8xb256-420e_coco-256x192.py

# Runtime settings
max_epochs = 50
base_lr = 5e-4
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
)

log_level = 'INFO'
load_from = None
resume = False

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=16)

# Codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# Model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa: E501
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=19,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# Custom 19-keypoint baseball metainfo
custom_metainfo = dict(
    dataset_name='baseball',
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper',
                swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper',
                swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper',
                swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper',
                swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper',
                swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper',
                swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper',
                swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper',
                swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper',
                swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper',
                 swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower',
                 swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower',
                 swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower',
                 swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower',
                 swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower',
                 swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower',
                 swap='left_ankle'),
        17: dict(name='bat_knob', id=17, color=[255, 255, 0], type='upper',
                 swap=''),
        18: dict(name='bat_barrel', id=18, color=[255, 255, 0], type='upper',
                 swap=''),
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_ankle', 'right_knee'), id=2,
                color=[255, 128, 0]),
        3: dict(link=('right_knee', 'right_hip'), id=3,
                color=[255, 128, 0]),
        4: dict(link=('left_hip', 'right_hip'), id=4,
                color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[0, 255, 0]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6,
                color=[255, 128, 0]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7,
                color=[51, 153, 255]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8,
                color=[0, 255, 0]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9,
                color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10,
                 color=[0, 255, 0]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11,
                 color=[255, 128, 0]),
        12: dict(link=('left_eye', 'right_eye'), id=12,
                 color=[51, 153, 255]),
        13: dict(link=('nose', 'left_eye'), id=13,
                 color=[51, 153, 255]),
        14: dict(link=('nose', 'right_eye'), id=14,
                 color=[51, 153, 255]),
        15: dict(link=('left_eye', 'left_ear'), id=15,
                 color=[51, 153, 255]),
        16: dict(link=('right_eye', 'right_ear'), id=16,
                 color=[51, 153, 255]),
        17: dict(link=('left_ear', 'left_shoulder'), id=17,
                 color=[0, 255, 0]),
        18: dict(link=('right_ear', 'right_shoulder'), id=18,
                 color=[255, 128, 0]),
        19: dict(link=('bat_knob', 'bat_barrel'), id=19,
                 color=[255, 255, 0]),
    },
    joint_weights=[1.0] * 19,
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.050, 0.050
    ],
)

# Base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/mmpose_baseball/'

backend_args = dict(backend='local')

# Pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.15,
        scale_factor=[0.6, 1.4],
        rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='mmdet.YOLOXHSVRandomAug',
        hgain=0.03,
        sgain=0.02,
        vgain=0.03),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.3,
                 contrast_limit=0.2,
                 p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=0,
                p=0.5),
            dict(type='GaussNoise', var_limit=(5.0, 5.0), p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=custom_metainfo,
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=custom_metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# Hooks
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(
        type='mmengine.hooks.EarlyStoppingHook',
        monitor='coco/AP',
        rule='greater',
        patience=10,
        min_delta=0.001),
]

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# NOTE: Temporal downsampling to simulate 15 fps should be handled in the
# dataset preprocessing / data loading stage via frame skipping before
# generating COCO annotations.
# NOTE: Mosaic / MixUp augmentation can be enabled by adding
# `mmdet.Mosaic` and `mmdet.YOLOXMixUp` transforms to the pipeline when
# the mmdet version in the environment supports them.
