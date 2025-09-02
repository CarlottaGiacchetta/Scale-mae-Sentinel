import torch
from torch.utils.data import DataLoader
import kornia.augmentation as K
from rasterio.enums import Resampling

from dataloaders.fmow_sentinel import FMOWSentinelDataset

# ==== Config basica ====
CSV_TRAIN = "WORK/fmoWSentinel/fmow-sentinel/train.csv"
ROOT_DIR  = "WORK/fmoWSentinel/fmow-sentinel"   # cartella base con split/train/...
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




dataset_train, sampler_train, train_collate = build_fmow_veg_sampler(
            args,
            config,
            transforms=None,
            num_replicas=num_tasks,
            rank=global_rank,
            transforms_init=transforms_init,
        )

# ==== Dataset + DataLoader ====
dataset = FMOWSentinelDataset(
    csv_path=CSV_TRAIN,
    root_dir=ROOT_DIR,
    split="train",
)
collate_fn = FMOWSentinelCollateFn(
        transforms=transforms,
        channels=[4, 5, 6],
        over_sample_factor=1.0,
        base_resolution=1.0,
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

# ==== Augmentazioni Kornia (batch-wise, GPU) ====
transforms_gpu = K.AugmentationSequential(
    K.RandomResizedCrop((224, 224), scale=(0.5, 1.0), resample="bicubic"),
    K.RandomHorizontalFlip(),
    K.Normalize(mean=torch.zeros(13), std=torch.ones(13)),  # placeholder per 13 bande
    data_keys=["input", "mask"],  # importante: gestisce sia img che mask
).to(DEVICE)

# ==== Loop di test ====
print(f"Uso device: {DEVICE}")

for batch in loader:
    imgs = batch["image"].to(DEVICE).float()          # (B,C,H,W)
    valid_masks = torch.ones_like(imgs, dtype=torch.uint8)

    print("Input shape:", imgs.shape)

    imgs_tgt, imgs_src, ratios, zero_ratio, valid_masks = transforms_gpu(imgs, valid_masks)

    print("After transforms:")
    print("  imgs_tgt:", imgs_tgt.shape)
    print("  imgs_src:", imgs_src.shape)
    print("  ratios:", ratios)
    break  # solo un batch di test
