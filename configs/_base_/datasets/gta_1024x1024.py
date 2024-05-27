gta_type = "CityscapesDataset"
gta_root = "data/gta/"
gta_crop_size = (1024, 1024)

gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(scale=(2560, 1440), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=gta_crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
gta_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2560, 1440), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_train_pipeline,
)

val_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_test_pipeline,
)