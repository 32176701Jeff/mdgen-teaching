def rmsf_geo_periods_to_npz_per_period(
    pdb_path,
    dcd_list,
    out_dir,
    period_list,
    stride=1,
    align_sel="protein and name CA",
    w1=27,
    chunk=1000,
):
    """
    純幾何 RMSF（Kabsch 對齊到 global frame 0）
    每個 period 輸出一個 npz
    """

    import os
    import numpy as np
    import mdtraj as md

    os.makedirs(out_dir, exist_ok=True)

    # ---------- Kabsch ----------
    def kabsch_rot(P, Q):
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return R

    def smooth(x, w):
        if w <= 1:
            return x.copy()
        return np.convolve(x, np.ones(w) / w, mode="same")

    def in_period(i, s, e):
        return s <= i < e

    # ---------- topology ----------
    top = md.load(pdb_path)
    ca_indices = top.topology.select(align_sel)
    if ca_indices.size == 0:
        raise ValueError(f"align_sel returned 0 atoms: {align_sel}")

    # ---------- containers ----------
    acc = {p: [] for p in period_list}

    ref_Qc = None
    ref_cQ = None
    global_frame = 0

    for dcd in dcd_list:
        for chunk_traj in md.iterload(dcd, top=pdb_path, chunk=chunk):
            xyz_chunk = chunk_traj.xyz[:, ca_indices, :].astype(np.float64)

            for i in range(xyz_chunk.shape[0]):

                if global_frame % stride != 0:
                    global_frame += 1
                    continue

                P = xyz_chunk[i]

                # reference = global frame 0
                if ref_Qc is None:
                    cQ = P.mean(axis=0)
                    ref_Qc = P - cQ
                    ref_cQ = cQ
                    global_frame += 1
                    continue

                for (s, e) in period_list:
                    if in_period(global_frame, s, e):
                        cP = P.mean(axis=0)
                        Pc = P - cP
                        R = kabsch_rot(Pc, ref_Qc)
                        aligned = Pc @ R + ref_cQ
                        acc[(s, e)].append(aligned)

                global_frame += 1

    # ---------- output per period ----------
    for (s, e), coords in acc.items():
        coords = np.asarray(coords)
        if coords.shape[0] == 0:
            print(f"[WARN] period {s}-{e}: no frames collected")
            continue

        mean = coords.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((coords - mean) ** 2, axis=2), axis=0))
        rmsf_s = smooth(rmsf, w1)

        out_npz = os.path.join(out_dir, f"rmsf_{s}_{e}.npz")
        np.savez(
            out_npz,
            ca_indices=ca_indices,
            period=np.array([s, e]),
            rmsf=rmsf,
            rmsf_smooth=rmsf_s,
            w1=w1,
        )



pdb = '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt.pdb'
dcd = ['/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt.dcd',
       '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt_0202.dcd',
       '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt_0203.dcd']
period = [
    (2000, 3000),
    (5000, 6000),
    (10000, 11000),
    (15000, 16000),]
npz_dir = '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/analysis/rmsf/npz'
rmsf_geo_periods_to_npz_per_period(pdb,dcd,npz_dir,period)