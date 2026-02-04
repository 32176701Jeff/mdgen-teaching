import argparse
import torch
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
args = parser.parse_args()

import os, torch, mdtraj, tqdm, time
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd




os.makedirs(args.out_dir, exist_ok=True)



def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}_fit.npy', 'r')

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)
    frames = atom14_to_frames(torch.from_numpy(arr))
    ####如果遇到沒有見過的胺基酸 就設為20
    seqres = torch.tensor([restype_order[c] if c in restype_order else 20 for c in seqres])
    ####如果遇到沒有見過的胺基酸 就設為20

    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}

    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
        
        
        
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres):

    item = get_batch(name, seqres, num_frames = model.args.num_frames)
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)  
    
    all_atom14 = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        # print(atom14[0,0,0,1], atom14[0,-1,0,1])
        all_atom14.append(atom14)

    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

    if args.xtc:
        traj = mdtraj.load(path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
        traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))


# 增加try erro, 避免一個name錯誤整個都不跑
import traceback
@torch.no_grad()
def main():
    model = NewMDGenWrapper.load_from_checkpoint(args.sim_ckpt)
    model.eval().to('cuda')

    df = pd.read_csv(args.split, index_col='name')

    failed = []
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        try:
            seq = df.loc[name, 'seqres']
            # 若 index 重複，loc 會回 Series；取第一個或自行處理
            if isinstance(seq, pd.Series):
                seq = seq.iloc[0]

            do(model, name, seq)

        except Exception as e:
            failed.append((name, repr(e)))
            print(f"[ERROR] name={name} failed: {e}")
            traceback.print_exc()
            # 避免 CUDA error 影響下一輪（尤其是 OOM 後）
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    if failed:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "failed_names.txt"), "w") as f:
            for n, err in failed:
                f.write(f"{n}\t{err}\n")
        print(f"Done. Failed {len(failed)} names. See failed_names.txt")

        

main()