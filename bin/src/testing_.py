def save_ca_xyz_3d_png(
    ca_xyz,
    out_path,
    connect=True,
    s=10,
    alpha=1.0,
    elev=20,
    azim=30,
    axis_range=None
):
    """
    將 Cα 三維座標畫成 3D 圖並輸出成 PNG（視角與範圍可固定）。

    參數：
    - ca_xyz: list 或 ndarray，形狀 (N, 3)
    - out_path: 輸出檔名
    - connect: 是否依序連線
    - s, alpha: 散點大小與透明度
    - elev, azim: 固定視角
    - axis_range: (xmin, xmax, ymin, ymax, zmin, zmax)，若為 None 則自動計算
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xyz = np.asarray(ca_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("ca_xyz 必須為形狀 (N, 3) 的座標陣列")

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    # 畫點與線
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, alpha=alpha)
    if connect and len(xyz) > 1:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1)

    # 固定範圍
    if axis_range is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = axis_range
    else:
        xmin, ymin, zmin = xyz.min(axis=0)
        xmax, ymax, zmax = xyz.max(axis=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # 固定視角
    ax.view_init(elev=elev, azim=azim)

    # 標籤
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path




import os
import subprocess
import shutil
import tempfile

def images_to_gif_ffmpeg(png_dir, output_gif, fps=10):
    """
    最簡單乾淨版：直接用 FFmpeg 將連續 PNG 合成 GIF。
    無 palettegen、無複雜濾鏡，適用所有版本。
    """

    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # 取得 PNG
    files = [f for f in os.listdir(png_dir) if f.endswith(".png")]
    if not files:
        raise ValueError(f"No PNG files found in {png_dir}")

    try:
        files_sorted = sorted(files, key=lambda x: float(os.path.splitext(x)[0]))
    except ValueError:
        files_sorted = sorted(files)

    # 建立暫時資料夾並重新命名成序列
    temp_dir = tempfile.mkdtemp(prefix="gif_")
    for i, name in enumerate(files_sorted):
        shutil.copy(os.path.join(png_dir, name),
                    os.path.join(temp_dir, f"{i:05d}.png"))

    pattern = os.path.join(temp_dir, "%05d.png")

    # ⚙️ 最基本的 FFmpeg 指令：不 palette、不花俏
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-vf", "scale=iw:ih:flags=bicubic",
        "-loop", "0",
        output_gif
    ]

    # 執行並顯示錯誤（若有）
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ FFmpeg stderr ↓↓↓")
        print(result.stderr)
        raise RuntimeError("FFmpeg command failed")

    shutil.rmtree(temp_dir)
    print(f"✅ GIF saved → {output_gif}")
    print("🧹 Temporary folder removed.")