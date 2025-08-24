# dataloaders/fmow_sentinel.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


import rasterio
from PIL import Image

from lib.transforms import get_inputs_outputs
from kornia.augmentation import AugmentationSequential

# ─────────────────────────────────────────────────────────
# Costanti fMoW-Sentinel2 (13 bande) — prese dal tuo codice
# ─────────────────────────────────────────────────────────
_FMOW_S2_MEAN = np.array([
    1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
    1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
    1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117
], dtype=np.float32)

_FMOW_S2_STD = np.array([
     633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
     948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
    1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813
], dtype=np.float32)

# Per sicurezza contro immagini giganti
Image.MAX_IMAGE_PIXELS = None

CATEGORIES = [
    "airport","airport_hangar","airport_terminal","amusement_park","aquaculture",
    "archaeological_site","barn","border_checkpoint","burial_site","car_dealership",
    "construction_site","crop_field","dam","debris_or_rubble","educational_institution",
    "electric_substation","factory_or_powerplant","fire_station","flooded_road","fountain",
    "gas_station","golf_course","ground_transportation_station","helipad","hospital",
    "impoverished_settlement","interchange","lake_or_pond","lighthouse","military_facility",
    "multi-unit_residential","nuclear_powerplant","office_building","oil_or_gas_facility",
    "park","parking_lot_or_garage","place_of_worship","police_station","port","prison",
    "race_track","railway_bridge","recreational_facility","road_bridge","runway","shipyard",
    "shopping_mall","single-unit_residential","smokestack","solar_farm","space_facility",
    "stadium","storage_tank","surface_mine","swimming_pool","toll_booth","tower",
    "tunnel_opening","waste_disposal","water_treatment_facility","wind_farm","zoo"
]

def _sentinel_vis_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    La tua "SentinelNormalize": clamp 2*std intorno al mean e rimappa a [0,255].
    Usata solo se vuoi preparare preview o debug; per il training usiamo Kornia.
    """
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (x - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class FMOWSentinelDataset(Dataset):
    """
    Dataset fMoW-Sentinel2:
    - Legge una CSV con colonne: category, location_id, image_id, timestamp (come nel tuo script)
    - Costruisce il path: {split}/{category}/{category}_{loc}/{category}_{loc}_{img}.tif
    - Ritorna un dict con:
        - "image": Tensor float32 (C,H,W), 13 canali, non normalizzata (normalizza la collate via Kornia)
        - "label": long (indice di categoria) — non usato nel pretrain MAE, ma utile per debug
        - "path" , "image_id", "timestamp"
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: str | None = None,
        split: str | None = None,
        years: list[int] | None = None,
        categories: list[str] | None = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        # Deduce split dal nome file se non specificato
        if split is None:
            name = os.path.basename(csv_path).lower()
            split = "train" if "train" in name else ("val" if "val" in name else "test")
        self.split = split

        self.image_size = 224

        # root_dir: se non passato, usa cartella accanto alla CSV "fmow-sentinel"
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(csv_path), "fmow-sentinel")
        self.root_dir = root_dir

        # Filtri opzionali
        if categories is not None and len(categories) > 0:
            # mantieni solo le righe con category in lista
            self.df = self.df[self.df["category"].isin(categories)]
            self.categories = categories
        else:
            self.categories = CATEGORIES

        if years is not None and len(years) > 0 and "timestamp" in self.df.columns:
            self.df["year"] = self.df["timestamp"].astype(str).str.slice(0, 4).astype(int)
            self.df = self.df[self.df["year"].isin(years)]

        # Indici stabili
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _build_image_path(self, row) -> str:
        cat = row["category"]
        loc = int(row["location_id"])
        img = int(row["image_id"])
        rel = f"{self.split}/{cat}/{cat}_{loc}/{cat}_{loc}_{img}.tif"
        return os.path.join(self.root_dir, rel)

    def _open_tif(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        # -> (H,W,C) float32
        return arr.transpose(1, 2, 0).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = self._build_image_path(row)
        np_img = self._open_tif(p)  # (H,W,C=13)
        # To torch (C,H,W) float32
        img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
        img = F.interpolate(img.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False).squeeze(0)


        y = int(CATEGORIES.index(row["category"])) if row["category"] in CATEGORIES else -1
        return {
            "image": img,  # (13,H,W) float32, nessuna normalizzazione qui
            "label": torch.tensor(y, dtype=torch.long),
            "image_id": int(row["image_id"]),
            "timestamp": row.get("timestamp", ""),
            "path": p,
        }

class FMOWSentinelCollateFn:
    """
    Collate compatibile con ScaleMAE:
    - Stacca batch (B,C,H,W)
    - Seleziona canali da config["data"]["channels"] (indici 0-based)
    - Applica pipeline Kornia (CustomCompose) con validmask fittizia
    - Produce (inputs, aux) come atteso da engine_pretrain/train_one_epoch:
        inputs = get_inputs_outputs(imgs_src, imgs_src_res, imgs, res)
        aux = dict(zero_ratio=..., valid_masks=...)
    """
    def __init__(
        self,
        transforms: AugmentationSequential,
        channels: list[int] | None,
        over_sample_factor: float,
        base_resolution: float,
    ):
        self.transforms = transforms
        self.channels = channels
        self.over_sample_factor = max(1.0, float(over_sample_factor))
        self.base_resolution = float(base_resolution)

    def __call__(self, samples: list[dict[str, torch.Tensor]]):
        # Stack immagini
        imgs = torch.stack([s["image"] for s in samples], dim=0)  # (B,C,H,W)
        B, C, H, W = imgs.shape

        # Selezione canali (se specificata in config)
        if self.channels is not None and len(self.channels) > 0:
            imgs = imgs[:, self.channels, :, :]
            C = imgs.shape[1]


        imgs = imgs.float()

        # valid mask piena (come shape (B,C,H,W) per matchare il codice esistente)
        valid_masks = torch.ones((B, C, H, W), dtype=torch.uint8, device=imgs.device)

        # Oversampling per batch più grande "logico" (come Sentinel2StackSampleCollateFn)
        tgt_b = int(B / self.over_sample_factor)
        if tgt_b <= 0:
            tgt_b = B
        # nessuna euristica zero_ratio: prendiamo i primi tgt_b
        imgs = imgs[:tgt_b].contiguous()
        valid_masks = valid_masks[:tgt_b].contiguous()

        # Applica pipeline Kornia (ritorna img_tgt, img_src, ratios, zero_ratio, valid_masks)
        imgs_tgt, imgs_src, ratios, zero_ratio, valid_masks = self.transforms(
            imgs, valid_masks
        )

        # risoluzione effettiva (ratio è crop_dim / original_dim -> res = ratio * base_res)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs_tgt.shape[-1] / imgs_src.shape[-1])

        # Confeziona inputs per MAE
        inputs = get_inputs_outputs(imgs_src, imgs_src_res, imgs_tgt, res)
        aux = dict(zero_ratio=zero_ratio, valid_masks=valid_masks)
        return inputs, aux

# ─────────────────────────────────────────────────────────
# Entry point per get_dataset_and_sampler(...)
# ─────────────────────────────────────────────────────────
def build_fmow_sentinel_sampler(config: dict, args, transforms, num_replicas: int, rank: int):
    """
    Usa i campi in config["data"]:

    Necessari:
      - csv_train: path alla CSV train
    Opzionali ma consigliati:
      - root_dir: directory base che contiene la struttura {split/category/...}.tif
      - channels: lista di indici canale da usare (0..12). Se assente usa tutti i 13.
      - oversample: fattore di oversampling per il batch (default=1)
    """
    dcfg = config["data"]
    csv_train = dcfg.get("csv_train", None)
    if csv_train is None:
        raise ValueError("config['data']['csv_train'] è richiesto per fMoW-Sentinel.")

    root_dir = dcfg.get("root_dir", None)
    channels = dcfg.get("channels", None)
    oversample = float(dcfg.get("oversample", 1.0))

    # Dataset
    dataset = FMOWSentinelDataset(
        csv_path=csv_train,
        root_dir=root_dir,
        split="train"
    )

    # Sampler distribuito standard (non geosampler)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    # Collate compatibile con il resto del framework
    collate_fn = FMOWSentinelCollateFn(
        transforms=transforms,
        channels=channels,
        over_sample_factor=oversample,
        base_resolution=getattr(args, "base_resolution", 1.0),
    )
    return dataset, sampler, collate_fn
