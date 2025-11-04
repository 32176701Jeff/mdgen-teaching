import sys
from pymol import cmd

pdb_file = sys.argv[-2]
out_png  = sys.argv[-1]

print(f"Loading {pdb_file}")
cmd.load(pdb_file, "protein")
cmd.hide("everything", "all")
cmd.show("cartoon", "protein")
cmd.color("cyan", "protein")
cmd.orient("protein")
cmd.ray(520, 520)
cmd.png(out_png, width=800, height=600, dpi=300, ray=0)
print(f"Saved to {out_png}")
cmd.quit()
