import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects

def plot_npy_contour(input_npy, output_png, sigma=2, cmap='viridis', fixed_extent=True):
    data = np.load(input_npy)

    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    elif data.ndim == 3:
        data = np.mean(data, axis=0)
    elif data.ndim != 2:
        raise ValueError(f"Unexpected shape: {data.shape}")

    data_smooth = gaussian_filter(data, sigma=sigma)
    vmin, vmax = np.nanmin(data_smooth), np.nanmax(data_smooth)
    extent = [0, data_smooth.shape[1], 0, data_smooth.shape[0]] if fixed_extent else None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    cf = ax.contourf(data_smooth, levels=50, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.contour(data_smooth, levels=15, colors='k', linewidths=0.25, extent=extent)

    ax.set_xlim(0, data_smooth.shape[1])
    ax.set_ylim(0, data_smooth.shape[0])
    ax.set_aspect('auto')
    ax.axis('off')

    cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.06, orientation='vertical')
    cbar.set_label('Intensity', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # 標題部分
    title_val = os.path.splitext(os.path.basename(input_npy))[0]
    try:
        title_val = f"{float(title_val):.2f}"
    except ValueError:
        pass
    t = ax.set_title(title_val, fontsize=13, fontweight='bold', pad=3)
    t.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                        path_effects.Normal()])  # 加黑邊讓字更清晰

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, pad_inches=0.05, dpi=300)
    plt.close(fig)













def rotate_trajectory(pdb_path, xtc_path, out_pdb, out_xtc, axis='z', angle_deg=90):
    import MDAnalysis as mda
    from MDAnalysis.transformations import rotate
    import numpy as np
    import os
    """
    對整個 trajectory 的所有 frame 進行 rigid rotation 並輸出新的 pdb 與 xtc。

    Parameters
    ----------
    pdb_path : str
        輸入 PDB 檔案路徑
    xtc_path : str
        輸入 XTC 檔案路徑
    out_pdb : str
        輸出 PDB 檔案路徑
    out_xtc : str
        輸出 XTC 檔案路徑
    axis : str
        旋轉軸 ('x', 'y', 'z')
    angle_deg : float
        旋轉角度（度）
    """

    # --- 1️⃣ 讀取系統 ---
    u = mda.Universe(pdb_path, xtc_path)
    ag = u.atoms

    # --- 2️⃣ 建立旋轉矩陣 ---
    theta = np.deg2rad(angle_deg)
    if axis == 'x':
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rot = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # --- 3️⃣ 定義 transformation pipeline ---
    def rigid_rotation(ts):
        com = ag.center_of_mass()
        coords = ag.positions - com
        ag.positions = np.dot(coords, rot.T) + com
        return ts

    u.trajectory.add_transformations(rigid_rotation)

    # --- 4️⃣ 輸出新 xtc ---
    os.makedirs(os.path.dirname(out_xtc), exist_ok=True)
    with mda.Writer(out_xtc, n_atoms=ag.n_atoms) as W:
        for ts in u.trajectory:
            W.write(ag)

    # --- 5️⃣ 輸出最後一幀的 pdb ---
    ag.write(out_pdb)

#============================================

def extract_ca_coordinates(pdb_file, xtc_file):
    import mdtraj as md
    import numpy as np
    """
    提取 PDB 和 XTC 檔案中的 C-α 原子座標（x, y, z），並準備 PCA 降維所需的數據。
    
    Parameters:
    pdb_file (str): PDB 檔案的路徑
    xtc_file (str): XTC 檔案的路徑
    
    Returns:
    np.ndarray: 每個時間步的 C-α 原子座標，形狀為 (n_frames, n_atoms * 3)
    """
    # 讀取 PDB 和 XTC 檔案
    traj = md.load(xtc_file, top=pdb_file)
    
    # 選擇 C-α 原子
    ca_atoms = traj.topology.select('name CA')
    
    # 提取所有時間步的 C-α 座標 (n_frames x n_atoms x 3)
    ca_coordinates = traj.xyz[:, ca_atoms, :]  # 形狀為 (n_frames, n_ca_atoms, 3)
    
    return ca_coordinates



def perform_pca(cas, n_components=2):
    from sklearn.decomposition import PCA
    import numpy as np
    """
    對所有的 C-α 原子座標數據進行 PCA，將每個 ca 降維到 2 個主成分。
    
    Parameters:
    cas (list of np.ndarray): 100 個 C-α 原子座標數據，每個形狀為 (n_frames, n_residues, 3)
    n_components (int): 要保留的主成分數量 (默認為 2)
    
    Returns:
    np.ndarray: 降維後的結果，形狀為 (100, 2)，每個 ca 只保留 2 個主成分
    """
    # 將所有 ca 數據組成一個大矩陣
    all_ca_reshaped = []

    for ca in cas:
        # 將每個 ca 數據展平為 (n_frames * n_residues * 3,)
        ca_flattened = ca.reshape(-1, ca.shape[-1])  # 每個 ca 都轉換為 (n_frames * n_residues, 3)
        ca_flattened = ca_flattened.flatten()  # 將每個 ca 展平成一維向量
        all_ca_reshaped.append(ca_flattened)
    
    # 將所有 ca 數據組成一個大矩陣，形狀為 (100, n_frames * n_residues * 3)
    all_ca_reshaped = np.vstack(all_ca_reshaped)
    
    # 執行 PCA 降維
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(all_ca_reshaped)
    
    # 返回降維後的結果，形狀為 (100, 2)，每個 ca 只保留 2 個主成分
    return pca_result



def mix_with_gauss(ca, t):
    import numpy as np
    """
    將 C-α 原子座標 (ca) 與高斯分佈 (gauss) 進行加權混合。
    
    Parameters:
    ca (np.ndarray): C-α 原子座標，形狀為 (n_frames, n_residues, 3)
    t (float): 用於控制 ca 與 gauss 的加權比例 (0 <= t <= 1)
    
    Returns:
    np.ndarray: 混合後的 C-α 原子座標，形狀為 (n_frames, n_residues, 3)
    """
    # 生成與 ca 同形狀的高斯隨機數據 (形狀為 (n_frames, n_residues, 3))
    gauss = np.random.normal(loc=0.0, scale=1.0, size=ca.shape)
    
    # 計算加權後的結果
    # gauss=x0
    # ca = x1
    # (1-t)=sigma t
    # t = alpha t
    # output=xt
    output = (1 - t) * gauss + t * ca
    
    return output


def mix_with_gauss_x_v(ca, t):
    import numpy as np
    """
    將 C-α 原子座標 (ca) 與高斯分佈 (gauss) 進行加權混合。
    
    Parameters:
    ca (np.ndarray): C-α 原子座標，形狀為 (n_frames, n_residues, 3)
    t (float): 用於控制 ca 與 gauss 的加權比例 (0 <= t <= 1)
    
    Returns:
    np.ndarray: 混合後的 C-α 原子座標，形狀為 (n_frames, n_residues, 3)
    """
    # 生成與 ca 同形狀的高斯隨機數據 (形狀為 (n_frames, n_residues, 3))
    gauss = np.random.normal(loc=0.0, scale=1.0, size=ca.shape)
    
    # 計算加權後的結果
    output_x = (1 - t) * gauss + t * ca
    output_v = -gauss + ca
    
    return output_x,output_v

#============================================