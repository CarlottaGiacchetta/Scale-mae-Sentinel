import os
import time
import math
import re
import numpy as np
import torch
from torch.utils.data import Sampler

import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample

from torch.utils.data.distributed import DistributedSampler
from lib.transforms import CustomCompose
from dataloaders.fmow_sentinel import PreprocessedChunks, FMOWSentinelCollateFn


BATCH_SIZE = 256                     # "base" usato per l'indicizzazione nei nomi file
DEVICE = "cpu"

over_sample_factor = float(1.5)
channels = [4, 5, 6]

# cartella di output (mantieni la tua)
ROOT_DIR = "/leonardo_scratch/fast/IscrC_UMC/preprocessed"

dataset = PreprocessedChunks(root=ROOT_DIR)
sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=False,
    )

# ==== Trasformazioni ====
other_transforms = AugmentationSequential(
    K.RandomHorizontalFlip(),
    K.Normalize(mean=[1263.73947144, 1645.40315151, 1846.87040806],
                std =[ 948.9819932 , 1108.06650639, 1258.36394548]),
)

transforms_train = CustomCompose(
    rescale_transform=K.RandomResizedCrop(
        (448, 448),
        ratio=(1.0, 1.0),
        scale=(0.2, 1.0),
        resample=Resample.BICUBIC.name,
    ),
    other_transforms=other_transforms,
    src_transform=K.Resize((224, 224)),
)

collate_fn = FMOWSentinelCollateFn(
    transforms=transforms_train,
    channels=channels,
    over_sample_factor=over_sample_factor,
    base_resolution=float(1.0),
)



# DataLoader
data_loader_train = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn,
)


total_items = len(dataset)


# ===========================
#   LOOP PRINCIPALE
# ===========================
n_seen = 0
t0 = time.time()

# NB: i parte da 0 qui, il batch globale è (resume_batch_i + i)
for i, batch in enumerate(data_loader_train):
    ((samples, res, targets, target_res), metadata) = batch
    print(f'Batch {i} /   samples.shape = {samples.shape}, res.shape={res.shape}, targets.shape={targets.shape}, target_res.shape={target_res.shape}')
    
    exit()
