_base_ = ['./pipelines/rand_aug.py']

# dataset settings
dataset_type = 'CustomDataset'
classes = ["No Gesture", "One Shot Needle Passing", "Pull The Suture", "Instrumental Tie", "Lay The Knot", "Cut The Suture"]  # The category names of your dataset

# dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[144.01279543590812, 131.91472349201618, 123.31423661654405], std=[64.7040393831716, 68.7886688576991, 70.7488325943194], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
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
        data_prefix='data/apas/rawframes',
        ann_file=f'data/apas/splits/train.split{split}.bundle',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/apas/rawframes',
        ann_file=f'data/apas/splits/val.split{split}.bundle',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/apas/rawframes',
        ann_file=f'data/apas/splits/test.split{split}.bundle',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# data = dict(
#     samples_per_gpu=64,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         data_prefix='data/imagenet/train',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_prefix='data/imagenet/val',
#         ann_file='data/imagenet/meta/val.txt',
#         pipeline=test_pipeline),
#     test=dict(
#         # replace `data/val` with `data/test` for standard test
#         type=dataset_type,
#         data_prefix='data/imagenet/val',
#         ann_file='data/imagenet/meta/val.txt',
#         pipeline=test_pipeline))

evaluation = dict(interval=10, metric='accuracy')
