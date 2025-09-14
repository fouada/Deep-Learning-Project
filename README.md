
# Chest X‑ray (Pneumonia vs Normal) — Project Overview & SAM Hyperparameters

This project trains image classifiers to distinguish **PNEUMONIA** vs **NORMAL** on the public Chest X‑ray dataset (Kaggle).  
It is built as a **modular training stack** with two small notebooks families (scratch vs. pretrained) and a shared `src/` codebase:

- **Data handling** – `DataCfg` (paths, image size, batch size, gray→RGB, balancing sampler, etc.).  
- **Training config** – `TrainCfg` (optimizer/scheduler knobs, early‑stopping, **SAM**, etc.).  
- **Models** – `build_custom_cnn`, `build_resnet18`, `build_vit_scratch`, `build_vit_pretrained`.  
- **Training/Eval** – `train_model`, `evaluate`, threshold search via `choose_threshold_by_min_recall` (target medical recall), plots and registry.

You can run:
- **Scratch**: `CustomCNN`, `ResNet‑18`, `ViT‑B/16` (with/without SAM).  
- **Pretrained**: `ResNet‑18` (ImageNet) and `ViT‑B/16` (ImageNet) fine‑tuned (with/without SAM).

> **Why SAM here?** Sharpness‑Aware Minimization (SAM) improves generalization by steering the optimizer away from sharp minima.  
> For **Transformers** (ViT) a **larger ρ** works best; for **CNN/ResNet** a **small ρ** is preferred.


---

## One‑table guide — TrainCfg knobs, purpose, and recommended values per model

> Values below match what ran stably in your notebooks on this dataset size. Use them as **defaults**; tune if you change batch size / augmentations.

| **Parameter** | **What it controls / Why** | **CustomCNN (scratch+SAM)** | **ResNet‑18 (scratch+SAM)** | **ResNet‑18 (pretrained+SAM)** | **ViT‑B/16 (scratch+SAM)** | **ViT‑B/16 (pretrained+SAM)** |
|---|---|---:|---:|---:|---:|---:|
| `epochs` | Max training epochs; early‑stopping ends sooner when `monitor` stalls. | **100** | **90** | **20** | **120** | **20** |
| `base_lr` | Base LR for optimizer; head LR = `base_lr × head_lr_mult` when used. | **3e‑4** | **3e‑4** | **1e‑4** | **3e‑4** | **1e‑4** |
| `weight_decay` | L2/AdamW decay; regularizes weights. | **1e‑4** | **1e‑4** | **1e‑4** | **1e‑4** | **1e‑4** |
| `label_smoothing` | Softens targets; can stabilize but may blur decision boundary; set 0 for clean comparison. | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| `use_sam` | Toggle Sharpness‑Aware Minimization. | **True** | **True** | **True** | **True** | **True** |
| `sam_rho` | SAM perturbation radius (ρ). **Small for CNN/ResNet**; **larger for ViT**. | **0.05** | **0.05** | **0.05** | **0.20** | **0.20** |
| `use_cosine` | Cosine LR decay after warmup. | **True** | **True** | **True** | **True** | **True** |
| `warmup_epochs` | LR warmup to avoid early instabilities. | **2** | **2** | **2** | **2** | **2** |
| `min_lr_mult` | Minimum LR as a fraction of `base_lr` (final LR = `min_lr_mult × base_lr`). | **0.05** | **0.05** | **0.05** | **0.05** | **0.05** |
| `monitor` | Metric to track for early‑stopping / best‑ckpt. Use AUROC for imbalanced medical data. | `"va_auroc"` | `"va_auroc"` | `"va_auroc"` | `"va_auroc"` | `"va_auroc"` |
| `patience` | Epochs with no improvement in `monitor` before stopping. | **15** | **15** | **5** | **15** | **5** |
| `clip_grad_norm` | Gradient global‑norm clip; prevents spikes. | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| `class_weights` | Per‑class weights for loss (handles imbalance); compute from train counts: `w_i = N/(K·(n_i+ε))`. | **use** | **use** | **use** | **use** | **use** |
| `llrd` | Layer‑wise LR decay for backbones (larger LR for head, smaller for deeper blocks). | **False** | **False** | **False** | **False** | **True** |
| `head_lr_mult` | Multiplier for classifier head LR; speeds head when fine‑tuning. | **1.0** | **1.0** | **10.0** | **1.0** | **10.0** |
| `use_amp` | Automatic Mixed Precision; speeds training if hardware supports. | **False** (safe) | **False** | **False** | **False** | **False** |

**Notes**
- The **ρ choices** reflect best practices: **ResNets/CNNs** favor **ρ≈0.02–0.05**, while **ViT‑B/16** benefits from **ρ≈0.2**.  
- Keep `patience=5` for pretrained fine‑tuning (backbone already near a good basin); use **longer patience** for scratch.
- If you enable **strong data augmentations**, you may reduce ViT’s `sam_rho` (e.g., ≈0.05–0.1) because augmentations already smooth the loss.
- `llrd=True` only for **ViT‑B/16 (pretrained)** in this repo; it’s unnecessary for scratch or for ResNet‑18 here.


---

## How to drop these into your notebooks

Where you create `TrainCfg`, paste the matching row (values) for the model you train, e.g.

```python
# ViT‑B/16 (pretrained + SAM)
pretrain_sam_cfg = TrainCfg(
    epochs=20, patience=5, monitor="va_auroc",
    base_lr=1e-4, weight_decay=1e-4,
    clip_grad_norm=1.0, use_amp=False,
    use_sam=True, sam_rho=0.20,
    use_cosine=True, warmup_epochs=2, min_lr_mult=0.05,
    target_recall=0.95,
    llrd=True, head_lr_mult=10.0,
    vit_npz_path=weights_path
)
```

…and similarly for the other entries from the table above. Keep your **`DataCfg` split** identical across runs for fair comparisons.

---

## Tuning tips

- **If validation loss/metrics oscillate early**: increase `warmup_epochs` → 3–5; for ViT try `sam_rho=0.15`.  
- **If recall < target after threshold search**: train longer (`epochs` +5), or increase `class_weights` skew (re‑compute after re‑split).  
- **If training slows**: turn `use_amp=True` on supported GPUs; reduce batch size only if you see OOM.  
- **If ViT underfits**: raise `head_lr_mult` → 15 and keep `llrd=True`.  
- **If ResNet overfits**: add small `label_smoothing=0.05` or mild augmentations.

---

### Provenance (why these defaults)

- ResNet/CNN use **small SAM ρ**; ViT uses **larger ρ**; this matches both your runs and common findings that **ViT‑B/16 benefits from ~0.2**, while **ResNets prefer ~0.02–0.05**.  
- Short **fine‑tuning schedules** with **larger head LR** converge quickly; **scratch** models need longer schedules and patience.

