# dataset settings
dataset_type = 'CustomDataset'
classes = ["No Gesture", "One Shot Needle Passing", "Pull The Suture", "Instrumental Tie", "Lay The Knot", "Cut The Suture"]  # The category names of your dataset

# dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

split = 3

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/apas/rawframes_with_keypoints',
        ann_file=f'data/apas/splits/train.split{split}.bundle',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/apas/rawframes_with_keypoints',
        ann_file=f'data/apas/splits/val.split{split}.bundle',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/apas/rawframes_with_keypoints',
        ann_file=f'data/apas/splits/test.split{split}.bundle',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
