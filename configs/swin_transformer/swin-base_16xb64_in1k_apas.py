_base_ = [
    '../_base_/models/swin_transformer/base_224_apas.py',
    '../_base_/datasets/imagenet_bs64_swin_224_apas.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth"