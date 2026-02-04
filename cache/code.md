## 1.Data representation and Data loader

### define chain_ids
- file : dataset.py
- function : MDGenDataset.__getitems__()
```python
        ...
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        ### ^^^
        chain_ids = np.concatenate([
            np.zeros(len_A, dtype=np.int64),
            np.ones(len_B, dtype=np.int64),
            2*np.ones(len_C, dtype=np.int64),])
        ### ^^^
        ...
```

### cropping
- file : dataset.py
- function : MDGenDataset.__getitems__()
```python
if self.args.atlas:
    if L > self.args.crop:
        start = np.random.randint(0, L - self.args.crop + 1) 
        
        torsions = torsions[:, start:start+self.args.crop]
        frames = frames[:, start:start+self.args.crop]
        seqres = seqres[start:start+self.args.crop]
        ### ^^^
        chain_ids = chain_ids[start:start+self.args.crop]
        ### ^^^
        mask = mask[start:start+self.args.crop]
        torsion_mask = torsion_mask[start:start+self.args.crop]

    elif L < self.args.crop:
        pad = self.args.crop - L
        ### ^^^
        chain_ids = np.concatenate([chain_ids, 3*np.ones(pad, dtype=np.int64)])
        ### ^^^
        ...
```

### chian information give to wrapper
- file : dataset.py
- function : MDGenDataset.__getitems__()
```python
...
return {
    'name': full_name,
    'frame_start': frame_start,
    'torsions': torsions,
    'torsion_mask': torsion_mask,
    'trans': frames._trans,
    'rots': frames._rots._rot_mats,
    'seqres': seqres,
    'mask': mask,
    ### ^^^
    'chain_ids': torch.from_numpy(chain_ids).long(), 
    ### ^^^
}
```

### give chain_ids to model
- file : wrapper.py
- function : NewMDGenWrapper.prep_batch()
prep_batch()
```python
...
return {
        'rigids': rigids,
        'latents': latents,
        'loss_mask': loss_mask,
        'model_kwargs': {
            'start_frames': rigids[:, 0],
            'end_frames': rigids[:, -1],
            'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
            'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
            'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
            'x_cond_mask': cond_mask,
            ### ^^^
            'chain_ids': batch['chain_ids'], 
            ### ^^^
        }
    }
```

## 2.chain embedding
### define chain_to_emb function
- file : latent_model.py
- function : latentMDGenModel.__init__()
```python
self.chain_to_emb = nn.Embedding(num_embeddings=4, embedding_dim=args.embed_dim)#有padding
```

### embedding aggregate
- file : latent_model.py 
- function : latetnMDGenModel.forward()
```python
def forward(self, x, t, mask, ..., chain_ids=None):
    ...
    x = self.latent_to_emb(x)
    if chain_ids is not None:
        chain_emb = self.chain_to_emb(chain_ids)
        chain_emb = chain_emb[:, None]
        x = x + chain_emb
```

## 3.fine tune ( step1 step2 step3)
### parameter setting
- file : parsing.py
```python
def parse_train_args():
    group = parser.add_argument_group("Optimization settings")
    
    group.add_argument("--lr", type=float, default=1e-3) #step1 step2 step3
    group.add_argument("--warmup_steps", type=int, default=0) #step1 step2 step3
    group.add_argument("--unfreeze_start_layer", type=int, default=10) #step2=6 step3=0
    group.add_argument("--unfreeze_all_dit", action='store_true', default=False) #step3 #2e-6 -> 2e-5
```

### wrapper
- file : wrapper.py
```python
class NewMDGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = LatentMDGenModel(args, self.latent_dim)

        # === #step1, #step2, #step3: Default freeze all parameters ===
        for name, p in self.model.named_parameters():
            p.requires_grad = False

        # === #step1, #step2, #step3: Always keep chain embedding trainable ===
        # Chain labels must be co-trained with any unfrozen layers
        for p in self.model.chain_to_emb.parameters():
            p.requires_grad = True

        # === #step1: Initialize weights ONLY at the beginning of the pipeline ===
        # We use unfreeze_start_layer=10 as a flag for the Alignment stage (Step 1)
        if self.args.unfreeze_start_layer == 10:
            # Small std ensures minimal disturbance to the pre-trained latent space
            torch.nn.init.normal_(self.model.chain_to_emb.weight, std=1e-3)

        # === #step2, #step3: Gradual Unfreezing of DiT Layers ===
        # Based on ULMFiT strategy to prevent catastrophic forgetting
        # Step 2: unfreeze_start_layer=6 (Layers 6-9)
        # Step 3: unfreeze_start_layer=0 (Layers 0-9)
        for i in range(self.args.unfreeze_start_layer, 10):
            for p in self.model.layers[i].parameters():
                p.requires_grad = True

    def configure_optimizers(self):
        # Only pass parameters with requires_grad=True to the optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )

        # === #step1, #step3: Linear Learning Rate Warm-up ===
        def lr_lambda(step):
            if self.args.warmup_steps > 0 and step < self.args.warmup_steps:
                # Linear ramp up from warmup_start_ratio to 1.0
                # Step 1: 0.0 -> 1.0 (Target: 1e-3)
                # Step 3: 0.1 -> 1.0 (Target: 2e-5, starts from 2e-6)
                alpha = float(step) / float(self.args.warmup_steps)
                return self.args.warmup_start_ratio + (1.0 - self.args.warmup_start_ratio) * alpha
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Trigger update at every training step
            },
        }
```

### train
- step1
```bash
python train.py --epochs 30 --lr 1e-3 --warmup_steps 4000 --unfreeze_start_layer 10 --warmup_start_ratio 0.0 --ckpt mdgen.ckpt
```
- step2
```bash
python train.py --epochs 20 --lr 5e-5 --unfreeze_start_layer 6 --warmup_steps 0 --ckpt path/to/step1_final.ckpt
```
- step3
```bash
python train.py --epochs 30 --lr 2e-5 --unfreeze_start_layer 0 --warmup_steps 8000 --warmup_start_ratio 0.1 --ckpt path/to/step2_final.ckpt
```