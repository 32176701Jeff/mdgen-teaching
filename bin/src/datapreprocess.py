
#============================================
def plot_ramachandran(md_pdb, md_xtc, save_path="rama_freq_white.png"):
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    u = mda.Universe(md_pdb, md_xtc)
    protein = u.select_atoms("protein and backbone")

    rama = Ramachandran(protein).run()
    angles = rama.angles  # shape = (n_frames, n_residues, 2)
    print("Shape of angles:", angles.shape)

    # 取出 phi / psi
    phi = angles[:, :, 0].ravel()
    psi = angles[:, :, 1].ravel()

    # 檢查單位
    print("phi range:", phi.min(), phi.max())
    if np.abs(phi).max() < 6.5:  # radians → degrees
        phi = np.degrees(phi)
        psi = np.degrees(psi)

    # 包角度到 [-180, 180)
    phi = ((phi + 180) % 360) - 180
    psi = ((psi + 180) % 360) - 180

    # 去除 NaN
    mask = ~np.isnan(phi) & ~np.isnan(psi)
    phi = phi[mask]
    psi = psi[mask]

    # 統計 normalized frequency
    bins = 180
    H, xedges, yedges = np.histogram2d(
        phi, psi,
        bins=[bins, bins],
        range=[[-180, 180], [-180, 180]],
        density=True
    )
    H = np.rot90(H)
    H = np.flipud(H)

    # === 設定 colormap：0 值 → 白色 ===
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_under('white')  # 對低於 vmin 的數值用白色
    cmap.set_bad('white')    # 對 NaN 用白色

    # 為了觸發 "under" 顏色，設個極小 vmin
    plt.figure(figsize=(6, 6))
    img = plt.imshow(
        H,
        extent=[-180, 180, -180, 180],
        cmap=cmap,
        origin='lower',
        aspect='auto',
        vmin=1e-8  # 讓 0 值顯示為白色
    )

    plt.xlabel("φ (phi) [°]")
    plt.ylabel("ψ (psi) [°]")
    plt.title("Ramachandran plot (normalized frequency)")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    cbar = plt.colorbar(img)
    cbar.set_label("Frequency (normalized 0–1)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()






#============================================
def calc_rmsd(md_pdb, md_xtc, save_path, atom_sel="protein"):
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    import matplotlib.pyplot as plt

    """
    計算並繪製 RMSD (Root Mean Square Deviation)
    
    參數：
        md_pdb (str): PDB 檔案路徑
        md_xtc (str): XTC 檔案路徑
        atom_sel (str): 原子選擇語法（預設為 'protein'）
        save_path (str): 輸出圖片檔案路徑（預設 'rmsd_plot.png'）
    """
    # --- 載入結構與軌跡 ---
    u = mda.Universe(md_pdb, md_xtc)
    ref = mda.Universe(md_pdb)  # 以初始構型作為對照
    
    print(f"Loaded trajectory with {len(u.trajectory)} frames")
    print(f"Atom selection: {atom_sel}")

    # --- 選擇對齊與計算的原子 ---
    mobile = u.select_atoms(atom_sel)
    reference = ref.select_atoms(atom_sel)

    # --- 計算 RMSD ---
    R = rms.RMSD(mobile, reference, select=atom_sel)
    R.run()

    time = R.rmsd[:, 1]  # 單位 ps
    rmsd_val = R.rmsd[:, 2]  # 單位 Å

    # --- 繪圖 ---
    plt.figure(figsize=(7, 4))
    plt.plot(time, rmsd_val, lw=1.5)
    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

#============================================
def plot_rmsf(pdb_file, xtc_file,out_png):
    import mdtraj as md
    import numpy as np
    import matplotlib.pyplot as plt
    """
    計算並輸出 C-α 原子 RMSF (Root Mean Square Fluctuation) 的結果，
    並將結果存成 npz 檔案。
    
    pdb_file (str): PDB 文件路径
    xtc_file (str): XTC 文件路径
    out_npz (str): 输出的 npz 文件路径
    """
    # 加载轨迹和拓扑
    traj = md.load(xtc_file, top=pdb_file)
    
    # 选择所有 C-α 原子
    calpha_atoms = traj.topology.select('name CA')
    
    # 计算 RMSF
    rmsf = md.rmsf(traj, traj, atom_indices=calpha_atoms)
    
    plt.plot(rmsf)
    plt.xlabel('Residue Index')
    plt.ylabel('RMSF (Å)')
    plt.grid(True)
    plt.savefig(out_png)
    plt.show()


#============================================
def modify_trajectory(pdb_file, xtc_file, output_pdb, output_xtc, frame_interval=40):
    import MDAnalysis as mda
    import numpy as np
    """
    修改轨迹时间步长，将每 10ps 的时间步长改为 400ps，
    并保存新的 PDB 和 XTC 文件。

    参数:
    pdb_file (str): 输入的 PDB 文件路径
    xtc_file (str): 输入的 XTC 轨迹文件路径
    output_pdb (str): 输出的 PDB 文件路径
    output_xtc (str): 输出的 XTC 文件路径
    frame_interval (int): 每隔多少帧保存一帧，默认每 40 帧（即 400ps）
    """
    # 加载 PDB 和 XTC 文件
    u = mda.Universe(pdb_file, xtc_file)
    
    # 创建一个新的 Universe 用于保存选中的帧
    with mda.Writer(output_xtc, u.atoms.n_atoms) as xtc_writer, mda.Writer(output_pdb, u.atoms.n_atoms) as pdb_writer:
        
        # 保存第一个帧作为 PDB 文件
        pdb_writer.write(u.atoms)

        # 选择每 40 帧（即 400ps）保存一帧
        for i, ts in enumerate(u.trajectory):
            if i % frame_interval == 0:
                # 写入选中的轨迹帧到新的 XTC 文件
                xtc_writer.write(u.atoms)


#============================================
def plot_ca_coordinates(pdb_file, output_png):
    import MDAnalysis as mda
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    """
    从 PDB 文件中提取 C-α 原子坐标并绘制 3D 散点图。
    
    参数:
    pdb_file (str): 输入的 PDB 文件路径
    output_png (str): 输出的 PNG 文件路径
    """
    # 加载 PDB 文件
    u = mda.Universe(pdb_file)
    
    # 选择所有 C-α 原子
    calpha_atoms = u.select_atoms("name CA")
    
    # 提取 C-α 原子的 3D 坐标
    coords = calpha_atoms.positions
    
    # 绘制 3D 散点图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', marker='o', s=50)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 保存图像为 PNG 文件
    plt.savefig(output_png)
    plt.show()

    print(f"3D coordinates plot saved to {output_png}")


#============================================
def protein_length_distribution(csv_file,out_png):
    import pandas as pd
    import matplotlib.pyplot as plt
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file)

    # 計算每個蛋白質的氨基酸長度
    df['seqres_length'] = df['seqres'].apply(len)

    # 畫出氨基酸長度的分佈圖
    plt.figure(figsize=(10, 6))
    plt.hist(df['seqres_length'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    

    plt.xlabel('Amino Acid Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(out_png)
    plt.show()


#============================================


def generate_residue_length_10_percent(input_csv, output_csv):
    import pandas as pd
    import numpy as np
    # 讀取 CSV 檔案
    df = pd.read_csv(input_csv)

    # 計算每個蛋白質的氨基酸長度（seqres_length）
    df['residue_length'] = df['seqres'].apply(len)

    # 隨機選擇 10% 的數據
    sample_size = int(len(df) * 0.1)  # 計算 10% 的大小
    sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)  # 隨機選擇 10% 的索引

    # 創建新的 DataFrame，只包含隨機選擇的 10% 蛋白質
    df_sampled = df.loc[sampled_indices]
    df_sampled.to_csv(output_csv, index=False)




