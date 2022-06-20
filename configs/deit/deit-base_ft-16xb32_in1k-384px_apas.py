_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_384_apas.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-base',
        img_size=384,
        patch_size=16,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=6,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    # Change to the path of the pretrained model
    # init_cfg=dict(type='Pretrained', checkpoint=''),
)

# data settings
data = dict(samples_per_gpu=32, workers_per_gpu=5)

load_from = "https://download.openmmlab.com/mmclassification/v0/deit/deit-base_3rdparty_ft-16xb32_in1k-384px_20211124-822d02f2.pth"