# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b8'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2816,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
