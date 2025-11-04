import sys
from pymol import cmd

# 用法：
# pymol -cq set_dihedral.pml -- input.pdb angle_deg output.png

pdb_file = sys.argv[-3]
angle = float(sys.argv[-2])  # 目標二面角度數
out_png = sys.argv[-1]

cmd.load(pdb_file, "mol")

# 假設 residue 2 是目標，抓出 atom id
# 這邊假設你的三個胺基酸的 atom name 是標準 PDB 命名
# C(1), N(2), CA(2), C(2)
cmd.select("C_prev", "resi 29 and name C")
cmd.select("N_j", "resi 30 and name N")
cmd.select("CA_j", "resi 30 and name CA")
cmd.select("C_j", "resi 30 and name C")

# 顯示該二面角（給出計算角度的四個原子）
cmd.dihedral("C_prev", "N_j", "CA_j", "C_j")

# 設定新的二面角
cmd.set_dihedral("C_prev", "N_j", "CA_j", "C_j", angle)

# 視覺化：將所有原子隱藏，並顯示 sticks 及球狀結構
cmd.hide("everything", "all")
cmd.show("sticks", "mol")
cmd.show("spheres", "C_prev or N_j or CA_j or C_j")  # 顯示球形結構
cmd.set("sphere_scale", 0.2, "C_prev or N_j or CA_j or C_j")
cmd.color("red", " N_j or CA_j ")  # 將這些原子顯示為紅色

# 輸出圖像
cmd.png(out_png, width=800, height=600, dpi=300, ray=1)
cmd.quit()

