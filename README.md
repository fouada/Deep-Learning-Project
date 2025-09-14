
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

# Hyperparameter tables

## SAM **ON**

| Model             | Init       |   epochs |   base_lr |   weight_decay |   label_smoothing | use_cosine   |   warmup_epochs |   min_lr_mult | monitor   |   patience |   clip_grad_norm | class_weights   | llrd   |   head_lr_mult | use_amp   |   sam_rho |   target_recall |
|:------------------|:-----------|---------:|----------:|---------------:|------------------:|:-------------|----------------:|--------------:|:----------|-----------:|-----------------:|:----------------|:-------|---------------:|:----------|----------:|----------------:|
| CustomCNN         | Scratch    |      100 |    0.1    |         0.001  |                 0 | True         |               3 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |              1 | False     |      0.02 |            0.95 |
| ResNet18          | Scratch    |       90 |    0.1    |         0.001  |                 0 | True         |               3 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |              1 | False     |      0.05 |            0.95 |
| ViT (scratch cfg) | Scratch    |      120 |    0.0003 |         0.3    |                 0 | True         |               5 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |             10 | False     |      0.2  |            0.95 |
| ResNet18          | Pretrained |       20 |    0.0001 |         0.0001 |                 0 | True         |               2 |          0.05 | va_auroc  |          5 |                1 | Yes             | False  |              1 | False     |      0.05 |            0.95 |
| ViT‑B/16          | Pretrained |       20 |    0.0001 |         0.0001 |                 0 | True         |               2 |          0.05 | va_auroc  |          5 |                1 | Yes             | True   |             10 | False     |      0.2  |            0.95 |

## SAM **OFF**

| Model             | Init       |   epochs |   base_lr |   weight_decay |   label_smoothing | use_cosine   |   warmup_epochs |   min_lr_mult | monitor   |   patience |   clip_grad_norm | class_weights   | llrd   |   head_lr_mult | use_amp   |   target_recall |
|:------------------|:-----------|---------:|----------:|---------------:|------------------:|:-------------|----------------:|--------------:|:----------|-----------:|-----------------:|:----------------|:-------|---------------:|:----------|----------------:|
| CustomCNN         | Scratch    |      100 |    0.1    |         0.001  |                 0 | True         |               3 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |              1 | False     |            0.95 |
| ResNet18          | Scratch    |       90 |    0.1    |         0.001  |                 0 | True         |               3 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |              1 | False     |            0.95 |
| ViT (scratch cfg) | Scratch    |      120 |    0.0003 |         0.3    |                 0 | True         |               5 |          0.01 | va_auroc  |         15 |                1 | Yes             | False  |             10 | False     |            0.95 |
| ResNet18          | Pretrained |       20 |    0.0001 |         0.0001 |                 0 | True         |               2 |          0.05 | va_auroc  |          5 |                1 | Yes             | False  |              1 | False     |            0.95 |
| ViT‑B/16          | Pretrained |       20 |    0.0001 |         0.0001 |                 0 | True         |               2 |          0.05 | va_auroc  |          5 |                1 | Yes             | True   |             10 | False     |            0.95 |

### Parameter quick reference
- **epochs** – total training epochs before early stopping.
- **base_lr** – initial learning rate used by the scheduler.
- **weight_decay** – L2 regularization (AdamW style for ViT).
- **label_smoothing** – softens hard labels; leave at 0.0 for medical binary tasks unless overconfident.
- **use_cosine** – cosine LR decay; smooth convergence.
- **warmup_epochs** – epochs to linearly ramp LR up from 0 → base_lr.
- **min_lr_mult** – LR floor as a fraction of base_lr in cosine schedule.
- **monitor** – validation metric for early stopping/best checkpoint.
- **patience** – epochs to wait with no improvement before stopping.
- **clip_grad_norm** – global gradient norm cap to stabilize training.
- **class_weights** – rebalance loss for class imbalance (Yes = use).
- **llrd** – layer-wise LR decay (usually only for ViT when fine‑tuning).
- **head_lr_mult** – LR multiplier for the classifier head (useful with llrd).
- **use_amp** – mixed precision; speeds up training on GPUs that support it.
- **sam_rho** – neighborhood radius for Sharpness‑Aware Minimization (only in *SAM ON* table).
- **target_recall** – recall used when choosing the operating threshold after training.


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

