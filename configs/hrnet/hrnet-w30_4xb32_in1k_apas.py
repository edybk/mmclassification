_base_ = [
    '../_base_/models/hrnet/hrnet-w30_apas.py',
    '../_base_/datasets/imagenet_bs32_pil_resize_apas.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w30_3rdparty_8xb32_in1k_20220120-8aa3832f.pth"