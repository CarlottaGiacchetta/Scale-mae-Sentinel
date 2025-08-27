# dataloaders/fmow_veg_geo.py
from __future__ import annotations
import os
from torch.utils.data.distributed import DistributedSampler

from dataloaders.fmow_sentinel import (
    FMOWSentinelDataset,
    FMOWSentinelCollateFn,
)

def _infer_csv_from_dir(dcfg: dict) -> str:
    """
    Permette due modi d'uso:
      - data.csv_train: path esplicito alla CSV
      - data.img_dir:   deduce <img_dir>/train.csv
    """
    csv_train = dcfg.get("csv_train") or dcfg.get("train_csv")
    if csv_train is None:
        img_dir = dcfg.get("img_dir") or dcfg.get("root_dir")
        if not img_dir:
            raise ValueError("Serve 'csv_train' oppure 'img_dir' in config['data'].")
        csv_train = os.path.join(os.path.expandvars(img_dir), "train.csv")
    return os.path.expandvars(csv_train)

def _common_build(channels_default: list[int], config: dict, args, transforms, num_replicas: int, rank: int):
    dcfg = config["data"]

    csv_train = _infer_csv_from_dir(dcfg)
    root_dir  = os.path.expandvars(dcfg.get("root_dir") or dcfg.get("img_dir") or os.path.dirname(csv_train))
    oversample = float(dcfg.get("oversample", 1.0))

    channels = dcfg.get("channels", channels_default)

    start_dataset = time.time()
    # Dataset
    dataset = FMOWSentinelDataset(
        csv_path=csv_train,
        root_dir=root_dir,
        split="train",
    )
    dataset_time = time.time() - start_dataset
    print(f'tempo caricamento dataset: {dataset_time:.2f} secondi')

    # Sampler DDP classico
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    # Collate compatibile con ScaleMAE (Kornia + get_inputs_outputs)
    collate_fn = FMOWSentinelCollateFn(
        transforms=transforms,
        channels=channels,
        over_sample_factor=oversample,
        base_resolution=getattr(args, "base_resolution", 1.0),
    )
    return dataset, sampler, collate_fn

def build_fmow_veg_sampler(config: dict, args, transforms, num_replicas: int, rank: int):
    """Profilo VEG: default channels = [4,5,6]"""
    return _common_build([4, 5, 6], config, args, transforms, num_replicas, rank)

def build_fmow_geo_sampler(config: dict, args, transforms, num_replicas: int, rank: int):
    """Profilo GEO: default channels = [7,10,11]"""
    return _common_build([7, 10, 11], config, args, transforms, num_replicas, rank)
