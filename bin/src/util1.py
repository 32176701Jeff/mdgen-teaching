import MDAnalysis as mda
from typing import List

def get_frame_count(psf_file, dcd_file):
    u = mda.Universe(psf_file, dcd_file)
    return len(u.trajectory)


def dcds_to_xtc(pdb: str, dcds: List[str], out_xtc: str):
    if len(dcds) == 0:
        raise ValueError("dcds list is empty")

    # Load universe with multiple trajectories
    u = mda.Universe(pdb, dcds)

    # Write concatenated XTC
    with mda.Writer(out_xtc, u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(u.atoms)

    print(f"✅ Written concatenated trajectory: {out_xtc}")


import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np

def xtc_to_rmsf_png(pdb, xtc, out_png,
                    selection="protein and backbone"):
    # Load system
    u = mda.Universe(pdb, xtc)

    # Atom selection
    ag = u.select_atoms(selection)
    if ag.n_atoms == 0:
        raise ValueError(f"No atoms selected with: {selection}")

    # RMSF calculation
    rmsf = rms.RMSF(ag).run()

    # Residue IDs (use resids for plotting)
    resids = ag.resids

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(resids, rmsf.rmsf, lw=1.5)
    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.title("Root Mean Square Fluctuation")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"✅ RMSF plot saved to: {out_png}")
