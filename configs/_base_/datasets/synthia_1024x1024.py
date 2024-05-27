syn_type = "CityscapesDataset"
syn_root = 'data/synthia/'
syn_crop_size = (1024, 1024)

syn_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(scale=(2560, 1440), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=syn_crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
syn_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2560,
        1580,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_syn = dict(
    type=syn_type,
    data_root=syn_root,
    data_prefix=dict(img_path='RGB', seg_map_path='GT/LABELS'),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=syn_train_pipeline,
)

val_syn = dict(
    ann_file='val.txt',
    data_prefix=dict(img_path='RGB', seg_map_path='GT/LABELS'),
    data_root='data/synthia/',
    img_suffix='.png',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            2560,
            1580,
        ), type='Resize'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='CityscapesDataset')
