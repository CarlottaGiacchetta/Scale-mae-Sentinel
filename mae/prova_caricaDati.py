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

from lib.transforms import CustomCompose
from dataloaders.fmow_sentinel import FMOWSentinelDataset, FMOWSentinelCollateFn


# ==== Config basica ====
CSV_TRAIN = "/leonardo_work/IscrC_UMC/fmoWSentinel/fmow-sentinel/train.csv"
ROOT_DIR  = "/leonardo_work/IscrC_UMC/fmoWSentinel/fmow-sentinel"   # cartella base con split/train/...
BATCH_SIZE = 1024                     # "base" usato per l'indicizzazione nei nomi file
DEVICE = "cpu"

over_sample_factor = float(1.0)
channels = [4, 5, 6]

# cartella di output (mantieni la tua)
OUTPUT_DIR = "/leonardo_scratch/fast/IscrC_UMC/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ==== Dataset ====
dataset = FMOWSentinelDataset(
    csv_path=CSV_TRAIN,
    root_dir=ROOT_DIR,
    split="train",
    channels=channels
)

# ==== Collate ====
collate_fn = FMOWSentinelCollateFn(
    transforms=transforms_train,
    channels=channels,
    over_sample_factor=over_sample_factor,
    base_resolution=float(1.0),
)

# ===========================
#   FUNZIONI DI SUPPORTO
# ===========================
def path_for_batch_index(i: int) -> str:
    """Ritorna il percorso del file per il batch 'i' secondo il naming corrente."""
    start = i * BATCH_SIZE
    end   = start + BATCH_SIZE
    return os.path.join(OUTPUT_DIR, f"preprocessed_{start}_{end}.npy")

def atomic_save_npy(arr: np.ndarray, out_path: str):
    """Salvataggio atomico: scrive su .tmp (file object) e poi rinomina su .npy."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    # scrivi su un file *aperto* così numpy NON aggiunge .npy al nome
    with open(tmp_path, "wb") as f:
        np.save(f, arr)      # nessun suffisso aggiunto
        f.flush()
        os.fsync(f.fileno()) # assicura che i dati siano su disco

    os.replace(tmp_path, out_path)


def compute_resume_batch_index() -> int:
    """
    Calcola l'indice batch da cui riprendere.
    Scorre da i=0 in su finché trova file continui validi:
      - se manca un file o l'ultimo è corrotto/non leggibile, riparte da lì.
    Questo gestisce anche buchi nel mezzo.
    """
    i = 0
    while True:
        p = path_for_batch_index(i)
        if not os.path.exists(p):
            # primo buco: riparto da qui
            return i
        # verifica veloce leggibilità (legge solo l'header con mmap)
        try:
            np.load(p, mmap_mode="r")
        except Exception as e:
            print(f"[RESUME] File presente ma non leggibile: {p} ({e}). Lo riscrivo.")
            return i
        i += 1

class RangeSampler(Sampler):
    """
    Sampler sequenziale con offset: restituisce indici [start, end) sull'intero dataset.
    Utile per saltare definitivamente gli esempi già processati.
    """
    def __init__(self, data_source_len: int, start: int = 0, end: int | None = None):
        self.data_len = data_source_len
        self.start = max(0, int(start))
        self.end = self.data_len if end is None else min(int(end), self.data_len)
        if self.start > self.end:
            self.start, self.end = self.end, self.end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


# ===========================
#   RESUME: CALCOLO OFFSET
# ===========================
# Numero di elementi che il DataLoader preleva ad ogni iterazione
loader_batch_size = int(BATCH_SIZE * over_sample_factor)

# Trova il primo batch "mancante"
resume_batch_i = compute_resume_batch_index()
# Quanti elementi del dataset sono già stati consumati?
# (assumiamo che ogni iterazione del DataLoader consumi 'loader_batch_size' elementi del dataset)
skip_items = resume_batch_i * loader_batch_size

if skip_items >= len(dataset):
    print(f"[RESUME] Tutto già processato: dataset={len(dataset)}, skip_items={skip_items}. Nulla da fare.")
    print("[DATASET-ONLY] Uscita.")
    raise SystemExit(0)

# Sampler sequenziale che parte direttamente dall'offset corretto
sampler = RangeSampler(len(dataset), start=skip_items, end=None)

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
remaining_items = len(sampler)  # quanti rimangono da processare da qui in avanti
estimated_remaining_batches = math.ceil(remaining_items / loader_batch_size)

print(
    "Modalità DATASET-ONLY con RESUME:",
    f"dataset totale={total_items}, già saltati={skip_items}, da processare ora={remaining_items},",
    f"batch stimati rimanenti ≈ {estimated_remaining_batches}"
)

# ===========================
#   LOOP PRINCIPALE
# ===========================
n_seen = 0
t0 = time.time()

# NB: i parte da 0 qui, il batch globale è (resume_batch_i + i)
for i, batch in enumerate(data_loader_train):
    global_batch_i = resume_batch_i + i
    start = global_batch_i * BATCH_SIZE
    end   = start + BATCH_SIZE

    out_path = path_for_batch_index(global_batch_i)
    if os.path.exists(out_path):
        # Per sicurezza: se esiste già, non riscrivo (può capitare in corner-case)
        print(f"[SKIP] Esiste già {out_path}. Salto batch globale {global_batch_i}.")
        continue

    (samples, res, targets, target_res), metadata = batch
    print(
        f"[batch {global_batch_i}] samples={tuple(samples.shape)} "
        f"res={tuple(res.shape)} "
        f"targets={'tensor'+str(tuple(targets.shape)) if torch.is_tensor(targets) else type(targets)} "
        f"target_res={tuple(target_res.shape)}"
    )

    # Conversione sicura su CPU
    arr = samples.detach().cpu().numpy()

    # Salvataggio atomico
    atomic_save_npy(arr, out_path)
    print(f"[OK] Salvato {out_path} | shape={arr.shape} ")

    n_seen += samples.size(0)
    if (i + 1) % 100 == 0:
        processed_items_est = (global_batch_i + 1) * loader_batch_size
        print(
            f"[PROGRESS] Batch locali {i+1}, batch globali {global_batch_i+1}, "
            f"Esempi processati (stima loader) {processed_items_est}/{total_items}"
        )

print(f"[DATASET-ONLY] Letti {n_seen} esempi in {time.time()-t0:.2f}s. Uscita.")
