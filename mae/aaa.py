# script una tantum per convertire i .npy memmap in fp16
import numpy as np, glob, os
src_root = "/leonardo_scratch/fast/IscrC_UMC/preprocessed"
for src in glob.glob(os.path.join(src_root, "*processed_*_*.npy")):
    arr = np.load(src, mmap_mode="r")
    dst = src.replace(".npy", ".fp16.npy")
    if not os.path.exists(dst):
        np.save(dst, arr.astype(np.float16, copy=False))
        print("->", dst)
