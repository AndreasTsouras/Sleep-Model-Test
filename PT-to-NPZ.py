import os
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def convert_pt_to_npz(pt_path, to_float16=False):
    npz_path = pt_path.replace('.pt', '.npz')
    
    if os.path.exists(npz_path):
        return f"⏩ Skipped: {os.path.basename(npz_path)}"
    
    tensor = torch.load(pt_path, map_location='cpu')
    
    if to_float16:
        tensor = tensor.half()

    np.savez_compressed(npz_path, data=tensor.numpy())
    return f"✅ Saved: {os.path.basename(npz_path)}"

def convert_all_joblib(folder_path, to_float16=False, n_jobs=32):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

    print(f"📁 Converting {len(files)} files in {folder_path} using {n_jobs} workers")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(convert_pt_to_npz)(f, to_float16) for f in tqdm(files, desc="🚀 Converting", unit="file")
    )

    print("✅ Done. Summary:")
    for r in results:
        print(r)
    
# Convert validation set to .npz (keep as float32)
convert_all_joblib('', to_float16=True, n_jobs=32) #add path with .pt files
