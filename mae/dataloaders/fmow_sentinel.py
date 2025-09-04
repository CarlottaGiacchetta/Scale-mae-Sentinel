# dataloaders/fmow_sentinel.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import time
import rasterio
from PIL import Image

from lib.transforms import get_inputs_outputs
from kornia.augmentation import AugmentationSequential

from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import Units

import os, re, glob, bisect
import numpy as np
import torch
from torch.utils.data import Dataset

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


_S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]

_SPAN_RE = re.compile(r'.*processed_(\d+)_(\d+)\.npy$')

def _parse_span(path: str) -> tuple[int, int]:
    m = _SPAN_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"Nome file non conforme: {path}")
    s, e = int(m.group(1)), int(m.group(2))
    return s, e

def _normalize_channels(channels):
    """
    Ritorna una lista di indici 1-based per rasterio (1..13).
    Accetta:
      - string preset: "rgb", "veg", "nir"
      - lista di int (0- o 1-based)
      - lista di string ("B04", "B03", ...)
    """
    if channels is None:
        # tutte le 13 bande
        return list(range(1, len(_S2_BANDS) + 1))


    idxs = []
    # prova a capire se sono stringhe o interi
    if all(isinstance(c, str) for c in channels):
        for name in channels:
            name = name.upper()
            if name not in _S2_BANDS:
                raise ValueError(f"Banda sconosciuta: {name}")
            idxs.append(_S2_BANDS.index(name) + 1)  # -> 1-based
        return idxs

    # qui sono int; gestisci 0-based o 1-based
    assert all(isinstance(c, int) for c in channels), "channels deve essere lista di int o nomi"
    mn, mx = min(channels), max(channels)
    if mn == 0 or mx <= len(_S2_BANDS) - 1:
        # sembra 0-based (0..12) -> +1
        idxs = [c + 1 for c in channels]
    else:
        # assume 1-based già (1..13)
        idxs = list(channels)

    # sanity check
    for c in idxs:
        if c < 1 or c > len(_S2_BANDS):
            raise ValueError(f"Indice banda fuori range (1..{len(_S2_BANDS)}): {c}")
    return idxs


class FMOWSentinelDataset(Dataset):
    """
    Legge solo le bande richieste da 'channels' usando rasterio.read(indexes=...).
    Ritorna 'image' in (C,H,W) float32, normalizzazione/resize delegati ai transforms.
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: str | None = None,
        split: str | None = None,
        years: list[int] | None = None,
        categories: list[str] | None = None,
        channels=None,  # <— NEW
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.band_idx = _normalize_channels(channels)  # <— indici 1-based per rasterio
        #self.data = None
        # deduci split da filename se non passato
        if split is None:
            name = os.path.basename(csv_path).lower()
            split = "train" if "train" in name else ("val" if "val" in name else "test")
        self.split = split
        

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(csv_path), "fmow-sentinel")
        self.root_dir = root_dir

        if categories:
            self.df = self.df[self.df["category"].isin(categories)]
            self.categories = categories
        else:
            self.categories = CATEGORIES

        if years and "timestamp" in self.df.columns:
            self.df["year"] = self.df["timestamp"].astype(str).str.slice(0, 4).astype(int)
            self.df = self.df[self.df["year"].isin(years)]

        self.df = self.df.reset_index(drop=True)
        #self.preload_data() #se usi Cache

    def __len__(self) -> int:
        return len(self.df)
    
    def preload_data(self):
        file_names = [self._build_image_path(row) for idx, row in self.df.iterrows()]
        from multiprocessing import Pool
        from tqdm import tqdm
        with Pool() as pool:
            self.cache = list(tqdm(pool.imap(self._read_bands, file_names), total=len(file_names)))
        print('CARICAMENTO FATTO ', len(self.cache), (len(file_names)))


    def _build_image_path(self, row) -> str:
        cat = row["category"]
        loc = int(row["location_id"])
        img = int(row["image_id"])
        rel = f"{self.split}/{cat}/{cat}_{loc}/{cat}_{loc}_{img}.tif"
        return os.path.join(self.root_dir, rel)

    def _read_bands(self, path: str) -> np.ndarray:
        # Leggi SOLO le bande richieste; rasterio usa 1-based indexes
        with rasterio.open(path) as src:
            arr = src.read(indexes=self.band_idx, out_dtype="float32")  # (C,H,W)
        return arr

    def __getitemCACHE__(self, idx: int):

        arr = self.cache[idx]
        img = torch.from_numpy(arr)  # (C,H,W) float32

        
        img = F.interpolate(img.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)

        y = int(CATEGORIES.index(row["category"])) if row["category"] in CATEGORIES else -1
        return {
            "image": img,           # (C_selected,H,W) float32
            "label": torch.tensor(y, dtype=torch.long),
            "image_id": int(row["image_id"]),
            "timestamp": row.get("timestamp", ""),
            "path": p,
        }

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = self._build_image_path(row)
        
        '''if self.data is None: 
            # (C,H,W) già float32 e nel giusto ordine; NO transpose/permute doppie
            self.data = self._read_bands(p)'''

        arr = self._read_bands(p)
        
        img = torch.from_numpy(arr)  # (C,H,W) float32
        img = F.interpolate(img.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)

        y = int(CATEGORIES.index(row["category"])) if row["category"] in CATEGORIES else -1
        return {
            "image": img,           # (C_selected,H,W) float32
            "label": torch.tensor(y, dtype=torch.long),
            "image_id": int(row["image_id"]),
            "timestamp": row.get("timestamp", ""),
            "path": p,
        }





class FMOWSentinelCollateFn:
    def __init__(self, transforms, channels, over_sample_factor, base_resolution: float):
        self.transforms = transforms
        self.over_sample_factor = int(over_sample_factor)
        self.base_resolution = float(base_resolution)
        # Prepara l’indice canali una volta sola (più veloce di advanced indexing)
        self.channel_idx = (torch.as_tensor(list(channels), dtype=torch.long)
                            if channels is not None else None)

    def __call__(self, samples):
        start_fwd = time.time()
        # 1) Stack veloce (evita stack_samples + dict juggling)
        imgs = torch.stack([s["image"] for s in samples], 0)  # (B,C,H,W)

        # 2) Selezione canali ottimizzata
        #if self.channel_idx is not None:
        #    imgs = imgs.index_select(1, self.channel_idx)

        B, C, H, W = imgs.shape

        # 3) Maschera compatta (B,1,H,W)
        valid_masks = torch.ones((B, 1, H, W), dtype=torch.uint8)

        # 4) Sotto-campionamento senza argsort su tensori enormi
        if self.over_sample_factor > 1:
            tgt_b = max(1, B // self.over_sample_factor)
            if "zero_ratio" in samples[0]:
                # Usa il valore già calcolato nel Dataset (molto più economico)
                zr = torch.tensor([s["zero_ratio"] for s in samples])
                order = torch.argsort(zr)[:tgt_b]
                imgs = imgs.index_select(0, order)
                valid_masks = valid_masks.index_select(0, order)
            else:
                # Fallback: taglia e basta (il tuo argsort precedente era no-op)
                imgs = imgs[:tgt_b]
                valid_masks = valid_masks[:tgt_b]

        # 5) Cast a float solo se serve ai transforms
        if imgs.dtype != torch.float32:
            imgs = imgs.float()

        # 6) Transforms (es. Kornia). Fai broadcast della mask solo qui se serve C
        if self.transforms is not None:
            m = valid_masks.expand(-1, C, -1, -1)  # broadcast lazily
            imgs, imgs_src, ratios, zero_ratio, m = self.transforms(imgs, m)
            res = ratios * self.base_resolution
            imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
            valid_masks = m

            return get_inputs_outputs(imgs_src, imgs_src_res, imgs, res), dict(
                zero_ratio=zero_ratio, valid_masks=valid_masks
            )

        # Fallback se non hai transforms (mantiene la stessa interfaccia)
        res = torch.full((imgs.size(0),), self.base_resolution, dtype=torch.float32)
        imgs_src = imgs
        imgs_src_res = res.clone()
        zero_ratio = torch.zeros((imgs.size(0),), dtype=torch.float32)
        valid_masks = valid_masks.expand(-1, C, -1, -1)
        time_collate = time.time() - start_fwd
        print('a fare la collate ci ha impiegato: ', time_collate)

        return get_inputs_outputs(imgs_src, imgs_src_res, imgs, res), dict(
            zero_ratio=zero_ratio, valid_masks=valid_masks
        )
 

class PreprocessedChunks(Dataset):
    def __init__(
        self,
        root: str,
        pattern: str = "*processed_*_*.npy",   # matcha anche 'preprocessed_*.npy'
        to_float32: bool = True,
        expand_2d: bool = True,
        mmap: bool = True,
    ):
        self.root = root
        self.files = sorted(
            glob.glob(os.path.join(root, pattern)),
            key=lambda p: _parse_span(p)[0]
        )
        print("numero di file trovati:", len(self.files))
        if not self.files:
            raise FileNotFoundError(f"Nessun file in {root} con pattern {pattern}")

        self.to_float32 = bool(to_float32)
        self.expand_2d = bool(expand_2d)
        self.mmap_mode = "r" if mmap else None

        # --- lunghezze reali per file (solo header, è veloce) ---
        self.lengths = []
        for fp in self.files:
            arr = np.load(fp, mmap_mode="r")  # non carica i dati, solo header
            n = int(arr.shape[0])
            self.lengths.append(n)
            del arr

        # indice cumulativo: starts[i] = indice globale di inizio del file i
        # es: [0, n0, n0+n1, ..., totale]
        self.starts = np.cumsum([0] + self.lengths).tolist()
        self.total_len = self.starts[-1]

        # cache del chunk corrente
        self._cur_file_idx = -1
        self.data = None

    def __len__(self) -> int:
        return int(self.total_len)

    def _load_chunk(self, file_idx: int):
        arr = np.load(self.files[file_idx], mmap_mode=self.mmap_mode)
        # (N,H,W) -> (N,1,H,W)
        if arr.ndim == 3 and self.expand_2d:
            arr = arr[:, None, :, :]
        # dtype
        if self.to_float32 and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        self.data = arr
        self._cur_file_idx = file_idx

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_len:
            raise IndexError

        # trova il file giusto: starts[file_idx] <= idx < starts[file_idx+1]
        file_idx = bisect.bisect_right(self.starts, idx) - 1
        in_file_idx = idx - self.starts[file_idx]

        if file_idx != self._cur_file_idx or self.data is None:
            self._load_chunk(file_idx)

        # slice dalla memmap; assicurati che sia contiguo prima di passare a torch
        x = np.ascontiguousarray(self.data[in_file_idx])
        x = torch.from_numpy(x)  # (C,H,W) o (1,H,W)

        return {
            "image": x,
            "label": -1,
            "image_id": int(idx),                 # id globale univoco
            "timestamp": "",
            "path": self.files[file_idx],
        }


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

    root_dir = dcfg.get("img_dir", None)
    channels = dcfg.get("channels", None)
    oversample = float(dcfg.get("oversample", 1.0))

    # Dataset
    dataset = FMOWSentinelDataset(
        csv_path=csv_train,
        root_dir=root_dir,
        split="train"
    )

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
