import os

import torch
import util.misc as misc
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import rasterio
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from .airound import AIROUND_DATASET_STATS
from .cvbrct import CVBRCT_DATASET_STATS
from .eurosat import EUROSAT_DATASET_STATS
from .fmow import FMOW_DATASET_STATS, build_fmow
from .imagelist import ImageList
from .imagenet100 import build_imagenet_sampler
from .mlrsnet import MLRSNET_DATASET_STATS
from .naip import build_naip_sampler
from .optimal import OPTIMAL_DATASET_STATS
from .resic45 import RESIC_DATASET_STATS, build_resic
from .sentinel2 import build_sentinel_sampler
from .ucmerced import UCMERCED_DATASET_STATS
from .whurs import WHURS_DATASET_STATS
from .xview import build_xview2_sampler

dataset_stats_lookup = {
    "airound": AIROUND_DATASET_STATS,
    "cvbrct": CVBRCT_DATASET_STATS,
    "mlrsnet": MLRSNET_DATASET_STATS,
    "resisc": RESIC_DATASET_STATS,
    "eurosat": EUROSAT_DATASET_STATS,
    "optimal-31": OPTIMAL_DATASET_STATS,
    "whu-rs19": WHURS_DATASET_STATS,
    "ucmerced": UCMERCED_DATASET_STATS,
    "fmow": FMOW_DATASET_STATS,
}
from dataloaders.fmow_veg_geo import (
    build_fmow_veg_sampler,
    build_fmow_geo_sampler,
)
from dataloaders.fmow_sentinel import build_fmow_sentinel_sampler

class FMOWSentinelEvalDataset(Dataset):
    """
    Dataset di valutazione per fMoW-Sentinel2 su 3 canali selezionati (VEG o GEO).
    Accetta:
      - path a una directory con struttura fMoW-Sentinel2 (split/category/...)
      - oppure path a un file lista (ImageList-style) con righe: "<path_to_tif> <label>"
      - oppure path a una CSV con colonne: path,label  (auto-detect semplice)

    Emissione: (tensor CxHxW float32 normalizzato, label int)
    """
    def __init__(self, root_or_list: str, channels: list[int], transform: transforms.Compose):
        self.channels = channels
        self.transform = transform

        p = os.path.expandvars(os.path.expanduser(root_or_list))
        self.samples = []

        if os.path.isdir(p):
            # Cammina la dir e costruisci (path,label) usando i nomi category -> label
            # mappa delle classi come negli altri dataset
            from dataloaders.fmow_sentinel import CATEGORIES  # riusa la lista già definita
            cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
            for cat in os.listdir(p):
                cat_dir = os.path.join(p, cat)
                if not os.path.isdir(cat_dir) or cat not in cat_to_idx:
                    continue
                lab = cat_to_idx[cat]
                for root, _, files in os.walk(cat_dir):
                    for f in files:
                        if f.lower().endswith(".tif"):
                            self.samples.append((os.path.join(root, f), lab))
        elif os.path.isfile(p):
            # Prova formato "path label" (ImageList)
            with open(p, "r", encoding="utf-8") as fh:
                head = fh.readline()
            if "," in head and "path" in head and "label" in head:
                # CSV minimale: path,label
                import csv
                with open(p, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        self.samples.append((os.path.expandvars(os.path.expanduser(row["path"])), int(row["label"])))
            else:
                # ImageList semplice
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        tifp = os.path.expandvars(os.path.expanduser(parts[0]))
                        lab = int(parts[1]) if len(parts) > 1 else -1
                        self.samples.append((tifp, lab))
        else:
            raise FileNotFoundError(f"Eval path non valido: {p}")

        if len(self.samples) == 0:
            raise RuntimeError(f"Nessun campione trovato in {p}")

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, path: str) -> torch.Tensor:
        # (C,H,W) float32
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        # seleziona canali 0-based
        arr = arr[self.channels, :, :]  # (C_sel,H,W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = self._read_tif(path)  # (C,H,W) float32 grezzo
        # La transform torchvision lavora anche su Tensor
        if self.transform is not None:
            x = self.transform(x)
        return x, label

def get_dataset_and_sampler(
    args,
    config,
    split="train",
    num_replicas=None,
    rank=None,
    transforms=None,
    transforms_init=None,
    linprobe_finetune=False,
):
    dataset_type = config["data"]["type"]
    print(dataset_type)
    if dataset_type == "NAIP":
        return build_naip_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "SENTINEL2":
        return build_sentinel_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "XView2":
        return build_xview2_sampler(
            config=config,
            num_replicas=num_replicas,
            rank=rank,
            transforms=transforms,
            split=split,
        )
    elif dataset_type == "ImageNet":
        return build_imagenet_sampler(
            config=config, num_replicas=num_replicas, rank=rank, transforms=transforms
        )
    elif dataset_type in ["fmow"]:
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"],
            transform=transforms_init,
            is_valid_file=is_fmow_rgb,
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type == "resisc":
        dataset = build_resic(config["data"]["img_dir"], transforms=transforms_init)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )
        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type == "eurosat":
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"], transform=transforms_init
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type in ("fmow_sentinel", "fmow-sentinel", "fmowsentinel"):
        return build_fmow_sentinel_sampler(config, args, transforms, num_replicas, rank)

    elif dataset_type in ("veg","VEG"):   # accetta anche type: VEG
        return build_fmow_veg_sampler(config, args, transforms, num_replicas, rank)

    elif dataset_type in ("geo","GEO"):   # accetta anche type: GEO
        return build_fmow_geo_sampler(config, args, transforms, num_replicas, rank)
    else:
        raise NotImplementedError


def is_fmow_rgb(fname: str) -> bool:
    return fname.endswith("_rgb.jpg")


class TransformCollateFn:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res), None


class TransformCollateFnLabel:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        labels = torch.tensor([x[1] for x in samples])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res, labels), None


def get_eval_dataset_and_transform(
    eval_dataset_id="resisc",
    eval_dataset_path="/leonardo_scratch/fast/IscrC_UMC/NWPU_RESISC45/NWPU-RESISC45",
    transforms_init=None,
    args=None,
):
    # All of these datasets are ImageFolders
    if eval_dataset_id in [
        "resisc",
        "mlrsnet",
        "airound",
        "cvbrct",
        "eurosat",
        "optimal-31",
        "whu-rs19",
        "ucmerced",
    ]:
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        transform_normalize = transforms.Normalize(
            mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
        )
        use_transforms = [transforms.ToTensor(), transform_normalize]
        if transforms_init:
            use_transforms.insert(0, transforms_init)
        if eval_dataset_id == 'ucmerced':
            use_transforms.insert(0, transforms.Resize((256,256)))
        transform_eval = transforms.Compose(use_transforms)

        if os.path.isdir(eval_dataset_path):
            dataset_eval = ImageFolder(eval_dataset_path, transform=transform_eval)
        else:
            print(eval_dataset_path)
            dataset_eval = ImageList(eval_dataset_path, transform=transform_eval)

    elif eval_dataset_id == "fmow":
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        if transforms_init and args:
            transform_eval = transforms.Compose(
                [
                    # Resize only the short side
                    transforms.Resize(args.eval_scale),
                    # TODO this may not be the right thing to do here.
                    transforms.CenterCrop(args.eval_scale),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        else:
            transform_eval = transforms.Compose(
                [
                    # TODO remove hardcoding px size?
                    transforms.Resize(512),  # downsample short side to 512
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        dataset_eval = build_fmow(eval_dataset_path, transforms=transform_eval)

    # ────────────── fMoW-Sentinel2 profilo VEG (B5,B6,B7 -> [4,5,6]) ──────────────
    elif eval_dataset_id_l in ["veg"]:
        # mean/std dalle tue stats sentinel (13-bande), slice sui canali usati
        # se vuoi, potresti anche caricarli da config, ma qui restiamo self-contained
        from dataloaders.fmow_sentinel import _FMOW_S2_MEAN as S2_MEAN, _FMOW_S2_STD as S2_STD  # type: ignore
        channels = [4, 5, 6]
        mean = torch.tensor(S2_MEAN[channels], dtype=torch.float32)
        std  = torch.tensor(S2_STD[channels], dtype=torch.float32)

        size = 224
        if args is not None:
            # args.eval_scale è una lista [56,112,224] o un int; prendiamo il max o l'int
            if isinstance(args.eval_scale, (list, tuple)):
                size = int(max(args.eval_scale))
            elif isinstance(args.eval_scale, int):
                size = args.eval_scale

        transform_eval = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ConvertImageDtype(torch.float32),  # in caso arrivi come uint16
            transforms.Normalize(mean=mean, std=std),
        ])
        dataset_eval = FMOWSentinelEvalDataset(
            root_or_list=eval_dataset_path,
            channels=channels,
            transform=transform_eval,
        )

    # ────────────── fMoW-Sentinel2 profilo GEO (B8,B11,B12 -> [7,10,11]) ──────────────
    elif eval_dataset_id_l in ["geo"]:
        from dataloaders.fmow_sentinel import _FMOW_S2_MEAN as S2_MEAN, _FMOW_S2_STD as S2_STD  # type: ignore
        channels = [7, 10, 11]
        mean = torch.tensor(S2_MEAN[channels], dtype=torch.float32)
        std  = torch.tensor(S2_STD[channels], dtype=torch.float32)

        size = 224
        if args is not None:
            if isinstance(args.eval_scale, (list, tuple)):
                size = int(max(args.eval_scale))
            elif isinstance(args.eval_scale, int):
                size = args.eval_scale

        transform_eval = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
        ])
        dataset_eval = FMOWSentinelEvalDataset(
            root_or_list=eval_dataset_path,
            channels=channels,
            transform=transform_eval,
        )

    else:
        raise NotImplementedError(f"Unknown eval dataset id: {eval_dataset_id}")

    return dataset_eval, transform_eval

    
