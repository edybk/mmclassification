_base_ = [
    '../_base_/models/resnet50_apas.py', '../_base_/datasets/apas_imagenet_bs32_split4.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth"
resume_from = None
workflow = [('train', 1)]
