from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

@torch.no_grad()
def _get_any(w, keys):
    for k in keys:
        if k in w: return w[k]
    raise KeyError(f"None of keys found: {keys}")

@torch.no_grad()
def load_jax_vit_npz_into_pytorch(model, npz_path: str | Path):
    """
    Load Google JAX ViT-B/16 checkpoint (.npz) into our PyTorch ViT.
    Supports both 'Mlp/Dense_0' and 'MlpBlock_3/Dense_0' naming.
    Keeps the PyTorch classifier head.
    """
    w = np.load(str(npz_path))
    D = model.head.in_features  # 768 for B/16

    def n2t(x): return torch.from_numpy(x).float()

    # Patch embed
    model.patch_embed.proj.weight.copy_(n2t(w['embedding/kernel']).permute(3,2,0,1))
    model.patch_embed.proj.bias.copy_(n2t(w['embedding/bias']))

    # CLS + pos embed (with interpolation)
    model.cls_token.copy_(n2t(w['cls']))
    posemb = w['Transformer/posembed_input/pos_embedding']
    if posemb.shape[1] != model.pos_embed.shape[1]:
        cls_pos = posemb[:, :1]
        grid_pos = posemb[:, 1:]
        gs_old = int(np.sqrt(grid_pos.shape[1]))
        grid_pos = grid_pos.reshape(1, gs_old, gs_old, D).transpose(0,3,1,2)
        H = W = int((model.patch_embed.num_patches) ** 0.5)
        grid_pos = F.interpolate(n2t(grid_pos), size=(H, W), mode='bicubic', align_corners=False)
        grid_pos = grid_pos.permute(0,2,3,1).reshape(1, H*W, D).numpy()
        posemb = np.concatenate([cls_pos, grid_pos], axis=1)
    model.pos_embed.copy_(n2t(posemb))

    # Blocks
    for i, blk in enumerate(model.blocks):
        blk.norm1.weight.copy_(n2t(_get_any(w, [f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])))
        blk.norm1.bias.copy_(n2t(_get_any(w,  [f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])))
        blk.norm2.weight.copy_(n2t(_get_any(w, [f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])))
        blk.norm2.bias.copy_(n2t(_get_any(w,  [f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])))

        # q,k,v
        Wq = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'])
        bq = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'])
        Wk = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'])
        bk = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'])
        Wv = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'])
        bv = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'])

        def flatten_qkv(W, b):
            if W.ndim == 3: W = W.reshape(W.shape[0], -1)  # [D, H*Hd]
            return W, b.reshape(-1)

        Wq, bq = flatten_qkv(Wq, bq); Wk, bk = flatten_qkv(Wk, bk); Wv, bv = flatten_qkv(Wv, bv)
        Wqkv = np.concatenate([Wq, Wk, Wv], axis=1).T
        bqkv = np.concatenate([bq, bk, bv], axis=0)
        blk.attn.qkv.weight.copy_(n2t(Wqkv)); blk.attn.qkv.bias.copy_(n2t(bqkv))

        # out proj
        Wo = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'])
        bo = _get_any(w, [f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        if Wo.ndim == 3: Wo = Wo.reshape(-1, Wo.shape[-1])
        blk.attn.proj.weight.copy_(n2t(Wo.T)); blk.attn.proj.bias.copy_(n2t(bo))

        # MLP — support both key schemes
        Wm0 = _get_any(w, [f'Transformer/encoderblock_{i}/Mlp/Dense_0/kernel',
                           f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'])
        bm0 = _get_any(w, [f'Transformer/encoderblock_{i}/Mlp/Dense_0/bias',
                           f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        Wm1 = _get_any(w, [f'Transformer/encoderblock_{i}/Mlp/Dense_1/kernel',
                           f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'])
        bm1 = _get_any(w, [f'Transformer/encoderblock_{i}/Mlp/Dense_1/bias',
                           f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])
        blk.mlp.fc1.weight.copy_(n2t(Wm0).T); blk.mlp.fc1.bias.copy_(n2t(bm0))
        blk.mlp.fc2.weight.copy_(n2t(Wm1).T); blk.mlp.fc2.bias.copy_(n2t(bm1))

    # Final LN (encoder_norm)
    model.norm.weight.copy_(n2t(w['Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(n2t(w['Transformer/encoder_norm/bias']))
    return model
