mol new /mnt/hdd/jeff/dataset/output/collagen/ser-0/raw/263_SER_1/raw/263_SER_1.psf
mol addfile /mnt/hdd/jeff/dataset/output/collagen/ser-0/raw/263_SER_1/npt-out/263_SER_1.dcd waitfor all

package require pbctools

set out_dir "/mnt/hdd/jeff/dataset/output/collagen/ser-0/raw/263_SER_1/postprocess"

set sel [atomselect top "protein"]

# PBC 處理（collagen 建議只 unwrap protein）
pbc center -sel "protein" -all
pbc unwrap -sel "protein" -all

# 寫 protein-only DCD
animate write dcd "$out_dir/protein.dcd" sel $sel waitfor all

# 寫 protein-only PDB（用 frame 0）
$sel frame 0
$sel writepdb "$out_dir/protein.pdb"

puts "Finished:"
puts "$out_dir/protein.dcd"
puts "$out_dir/protein.pdb"

exit
