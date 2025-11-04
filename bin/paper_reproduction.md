[README.md](../README.md)
## conda install
>```
>pip install numpy==1.21.2 pandas==1.5.3
>pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
>pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
>pip install wandb dm-tree einops torchdiffeq fair-esm pyEMMA
>pip install matplotlib==3.7.2 numpy==1.21.2
>```

## download ATLAS
Download the ATLAS simulations via https://github.com/bjing2016/alphaflow/blob/master/scripts/download_atlas.sh to data/atlas_sims

## preprocess, making .npy
>```
># Prep with interval 40 * 10 ps = 400 ps
>python -m scripts.prep_sims --splits splits/atlas.csv --sim_dir data/atlas_sims --outdir data/atlas_data --num_workers [N] --suffix _i40 --stride 40
>```

## checkpoint
>```
>wget https://storage.googleapis.com/mdgen-public/weights/atlas.ckpt
>```

## inference
>```
># ATLAS forward simulation # note no --xtc here!
>python sim_inference.py --sim_ckpt atlas.ckpt --data_dir share/data_atlas/ --num_frames 250 --num_rollouts 1 --split splits/atlas_test.csv --suffix _R1 --out_dir [DIR]
>```

## evaluation
要去下載analyze_ensembles.py

To analyze the ATLAS rollouts, follow the instructions at https://github.com/bjing2016/alphaflow?tab=readme-ov-file#Evaluation-scripts.

這是我去上面下載下來的檔案
[analyze_ensembles.py](../scripts/analyze_ensembles.py)