import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

def save_lastframes_ca_to_npy(pdbxtc_dir, out_npy_dir, last_n, stride=10, max_proteins=None):
    """
    讀 pdbxtc_dir 內同名 aaa.pdb/aaa.xtc
    每個 protein 取最後 last_n frames 的 CA 座標，存成 out_npy_dir/aaa.npy
    aaa.npy shape: (n_frames_kept, n_ca, 3) dtype=float32
    """
    os.makedirs(out_npy_dir, exist_ok=True)

    pdbs, xtcs = {}, {}
    for fn in os.listdir(pdbxtc_dir):
        path = os.path.join(pdbxtc_dir, fn)
        if fn.lower().endswith(".pdb"):
            pdbs[os.path.splitext(fn)[0]] = path
        elif fn.lower().endswith(".xtc"):
            xtcs[os.path.splitext(fn)[0]] = path

    names = sorted(set(pdbs.keys()) & set(xtcs.keys()))
    if max_proteins is not None:
        names = names[:max_proteins]
    if len(names) == 0:
        raise RuntimeError(f"No matched *.pdb/*.xtc pairs found in: {pdbxtc_dir}")

    for name in names:
        pdb = pdbs[name]
        xtc = xtcs[name]
        out_npy = os.path.join(out_npy_dir, f"{name}.npy")

        try:
            u = mda.Universe(pdb, xtc)
            ca = u.select_atoms("name CA")
            if ca.n_atoms == 0:
                print(f"[SKIP] {name}: no CA")
                continue

            frame_indices = list(range(0, last_n, stride))

            arr = np.empty((len(frame_indices), ca.n_atoms, 3), dtype=np.float32)
            for i, fi in enumerate(frame_indices):
                u.trajectory[fi]
                arr[i] = ca.positions.astype(np.float32, copy=False)

            np.save(out_npy, arr)
            print(f"[OK] saved {out_npy}  shape={arr.shape}")

        except Exception as e:
            print(f"[FAIL] {name}: {e}")


pdbxtc = '/mnt/hdd/jeff/dataset/output/collagen/zh-all-pdbxtc'
npy = '/mnt/hdd/jeff/dataset/output/collagen/zh-all/mdgen-collagen-lidar/npy'
last_n = 90
save_lastframes_ca_to_npy(pdbxtc,npy,last_n)