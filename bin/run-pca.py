def pca_kabsch_stream_to_npz(
    pdb_path,
    dcd_list,
    out_npz,
    align_sel="protein and name CA",
    n_components=2,
    ref_frame_global=0,
    chunk=1000,
):
    """
    Streaming PCA with Kabsch alignment
    - mdtraj.iterload (RAM-safe)
    - IncrementalPCA
    - output npz (no plotting)
    """

    import numpy as np
    import mdtraj as md
    from sklearn.decomposition import IncrementalPCA

    # ---------- Kabsch ----------
    def kabsch_rot(P, Q):
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return R

    def align_to_ref(P, Qc, cQ):
        cP = P.mean(axis=0)
        Pc = P - cP
        R = kabsch_rot(Pc, Qc)
        return Pc @ R + cQ

    # ---------- normalize dcd_list ----------
    if isinstance(dcd_list, str):
        dcd_list = [dcd_list]

    # ---------- topology ----------
    top = md.load(pdb_path)
    atom_idx = top.topology.select(align_sel)
    if atom_idx.size == 0:
        raise ValueError(f"align_sel returned 0 atoms: {align_sel}")

    N = len(atom_idx)
    ipca = IncrementalPCA(n_components=n_components)

    # ---------- pass 1: partial_fit ----------
    ref_Qc = None
    ref_cQ = None
    global_frame = 0

    print("PCA pass 1: fitting IncrementalPCA")

    for dcd in dcd_list:
        for chunk_traj in md.iterload(dcd, top=pdb_path, chunk=chunk):
            xyz = chunk_traj.xyz[:, atom_idx, :].astype(np.float64)

            X_batch = []

            for i in range(xyz.shape[0]):
                P = xyz[i]

                if ref_Qc is None:
                    if global_frame == ref_frame_global:
                        cQ = P.mean(axis=0)
                        ref_Qc = P - cQ
                        ref_cQ = cQ
                    global_frame += 1
                    continue

                aligned = align_to_ref(P, ref_Qc, ref_cQ)
                X_batch.append(aligned.reshape(-1))
                global_frame += 1

            if X_batch:
                ipca.partial_fit(np.vstack(X_batch))

    # ---------- pass 2: transform ----------
    print("PCA pass 2: transforming coordinates")

    Z_all = []
    frame_order = []

    global_frame = 0
    ref_Qc = None
    ref_cQ = None

    for dcd in dcd_list:
        for chunk_traj in md.iterload(dcd, top=pdb_path, chunk=chunk):
            xyz = chunk_traj.xyz[:, atom_idx, :].astype(np.float64)

            X_batch = []

            for i in range(xyz.shape[0]):
                P = xyz[i]

                if ref_Qc is None:
                    if global_frame == ref_frame_global:
                        cQ = P.mean(axis=0)
                        ref_Qc = P - cQ
                        ref_cQ = cQ
                    global_frame += 1
                    continue

                aligned = align_to_ref(P, ref_Qc, ref_cQ)
                X_batch.append(aligned.reshape(-1))
                frame_order.append(global_frame)
                global_frame += 1

            if X_batch:
                Z = ipca.transform(np.vstack(X_batch))
                Z_all.append(Z)

    Z_all = np.vstack(Z_all)
    frame_order = np.array(frame_order)

    # ---------- save ----------
    np.savez(
        out_npz,
        pc_coords=Z_all,
        frame_order=frame_order,
        evr=ipca.explained_variance_ratio_,
        align_sel=align_sel,
        n_atoms=N,
    )

    print(f"PCA data saved to: {out_npz}")

pdb = '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt.pdb'
dcd = ['/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt.dcd',
       '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt_0202.dcd',
       '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/raw/wt_0203.dcd']
npz = '/mnt/hdd/jeff/dataset/output/collagen/namd/wt/analysis/pca/npz/0-25000.npz'

pca_kabsch_stream_to_npz(
    pdb,
    dcd,
    npz,
)