_base_ = [
    '../_base_/models/convnext/convnext-large_apas.py',
    '../_base_/datasets/imagenet_bs64_swin_224_apas.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=64)

optimizer = dict(lr=4e-3)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

load_from = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth"