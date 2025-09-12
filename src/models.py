"""
Models:
- CustomCNN (small): a light baseline for ablations (scratch only)
- ResNet18: scratch & pretrained (torchvision)
- ViT (scratch): minimal ViT (patch embed + encoder)
- ViT (pretrained): via timm OR Google .npz fallback
"""
from __future__ import annotations
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models as tv_models

# --------- Custom CNN (scratch baseline) ---------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CustomCNN(nn.Module):
    """
    Small CNN for ablations:
      Stem: 3x3 conv
      Stages: (32)->(64)->(128)
      GlobalAvgPool + 2-way head
    ~0.1–0.5M params depending on width.
    """
    def __init__(self, num_classes=2, width=32, dropout=0.0):
        super().__init__()
        self.stem = ConvBlock(3, width, k=3, s=1, p=1)
        self.stage1 = nn.Sequential(ConvBlock(width, width), ConvBlock(width, width))
        self.down1  = ConvBlock(width, width*2, s=2)
        self.stage2 = nn.Sequential(ConvBlock(width*2, width*2), ConvBlock(width*2, width*2))
        self.down2  = ConvBlock(width*2, width*4, s=2)
        self.stage3 = nn.Sequential(ConvBlock(width*4, width*4), ConvBlock(width*4, width*4))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width*4, num_classes)
        )
        self.num_classes = num_classes
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.down1(x)
        x = self.stage2(x); x = self.down2(x)
        x = self.stage3(x)
        return self.head(x)

def build_custom_cnn(num_classes=2, width=32, dropout=0.0) -> nn.Module:
    return CustomCNN(num_classes=num_classes, width=width, dropout=dropout)

# --------- ResNet18 ---------
def build_resnet18(in_chans=3, num_classes=2, pretrained=False, conv_stem=False) -> nn.Module:
    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_chans != 3:
        # replace stem conv to accept non-3 channels (kept 3 in data)
        m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if conv_stem:
        # optional: 3x3 conv stem
        m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.num_classes = num_classes
    return m

# --------- Minimal ViT (scratch) ---------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size // patch) * (img_size // patch)
    def forward(self, x):
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=6, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each: [B,N,H,Ch]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=2,
                 embed_dim=384, depth=12, heads=6, patch=16, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, :N+1]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

def build_vit_scratch(img_size=224, in_chans=3, num_classes=2, embed_dim=384, depth=12, heads=6, patch=16, mlp_ratio=4.0, drop=0.1):
    return VisionTransformer(img_size, in_chans, num_classes, embed_dim, depth, heads, patch, mlp_ratio, drop)

# --------- ViT-B/16 pretrained (timm preferred, npz fallback) ---------
def build_vit_pretrained(num_classes=2, model_name="vit_base_patch16_224", pretrained=True, npz_path: Optional[str] = None):
    """
    Preferred: timm.create_model('vit_base_patch16_224', pretrained=True)
    Fallback: if npz_path provided, build compatible ViT-B/16 and load Google .npz weights.
    """
    try:
        import timm
        if pretrained:
            m = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            m = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        m.num_classes = num_classes
        return m
    except Exception:
        pass
    if npz_path is not None:
        from .vit_npz_loader import load_jax_vit_npz_into_pytorch
        m = VisionTransformer(img_size=224, in_chans=3, num_classes=num_classes,
                              embed_dim=768, depth=12, heads=12, patch=16, mlp_ratio=4.0, drop=0.0)
        m = load_jax_vit_npz_into_pytorch(m, npz_path)
        return m
    raise RuntimeError("ViT pretrained requested but neither timm nor npz weights are available.")
