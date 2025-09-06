# EXPERIMENTS.md — ViT Chest X‑ray (Pneumonia) 

This document records the **experiments, results, and lessons** from building and evaluating two Vision Transformer (ViT) variants on the **Chest X‑ray Pneumonia** dataset:

1. **Custom ViT (from scratch)** — faithful to the ViT paper design, plus light regularization (DropPath, label smoothing).  
2. **Pretrained ViT‑B/16** — initialized from **Google’s JAX `.npz` checkpoint** (offline), loaded into an equivalent PyTorch ViT skeleton and fine‑tuned.

The project targets a **recall‑first** operating point suitable for screening (catch as many *PNEUMONIA* cases as possible, even at a small precision cost).

---

## Dataset & Splits

- Source: Kaggle **Chest X‑Ray Images (Pneumonia)** (`paultimothymooney/chest-xray`)
- Folder layout expected by the notebook: `Data/chest_xray/{train,val,test}/...`
- **Important:** the official `val/` folder has only **16** images (unstable).  
  We **re‑split** the original **train/** into:
  - **train_ds** and **val_ds** with a **stratified 20% validation** split
  - **test_ds** uses the official Kaggle **test/** as provided
- Final split used in the best runs:
  - `len(train_ds), len(val_ds), len(test_ds) = (4172, 1044, 624)`
  - Class balance in **train_ds** printed at runtime (e.g., `[1073 NORMAL, 3099 PNEUMONIA]`)

---

## Metrics & Operating Points

- Metrics: **Accuracy, Precision, Recall, F1, AUROC** (macro decision metrics; AUROC threshold‑independent).
- We report two operating points:
  - **@0.5** — fixed probability threshold = 0.5 (baseline view).
  - **@Rec** — **recall‑first threshold** calibrated on **VAL** to achieve **recall ≥ 0.95** with the **highest possible precision**. That threshold is then applied to **TEST**.

Calibration helper functions in the notebook:
- `choose_threshold_by_min_recall(y_val, p1_val, min_recall=0.95)` → `(thr_best, recall_at_thr, precision_at_thr)`  
- `summarize_at_threshold(y, p1, thr)` → dict with `acc, precision, recall, f1, auroc, thr`

---

## Models

### A) Custom ViT (from scratch)
- **Input**: `224×224`, from grayscale X‑ray converted to **3 channels** for compatibility.
- **Patchify**: `16×16` conv with stride 16 → `N=196` patches.
- **Tokens**: prepend **[CLS]**, add **learnable 1‑D positional embeddings**.
- **Blocks**: **Pre‑LN** design (LN → MSA → residual, LN → MLP(GELU) → residual); **depth=6**, **heads=8**, **embed_dim=256**, **mlp_ratio=4**.
- **Regularization**: dropout + **DropPath** (stochastic depth, global rate ~`0.05` linearly scheduled per depth).  
- **Loss**: Cross‑Entropy + **label smoothing** (`0.05`) with optional **class weights**.  
- **Optim**: **AdamW** (`lr=3e-4`, `wd=0.05`), **early stopping** on **val AUROC** (`patience=5`).

> **DropPath** module is implemented in the notebook and inserted inside the residual branches:  
> `x = x + drop_path(attn(norm1(x)))` and `x = x + drop_path(mlp(norm2(x)))`.

### B) Pretrained ViT‑B/16 (offline)
- **Skeleton**: PyTorch ViT with **embed_dim=768, depth=12, heads=12, mlp_ratio=4**.
- **Weights**: offline **Google JAX `.npz`** (`notebooks/weights/ViT-B_16.npz`).  
  A custom loader maps:
  - Patch embedding `[P,P,in,D] → [D,in,P,P]`
  - Attention **q/k/v** DenseGeneral → merged **qkv** linear
  - Out projection `[H,Hd,D] → [D_out,D_in]`
  - MLP Dense_0 / Dense_1 → PyTorch Linear
  - LayerNorm scales/biases
  - Positional embeddings with **grid interpolation** if needed
- **Head**: keep a **2‑class** linear classifier; fine‑tune with **AdamW** (`lr=1e-4`, `wd=0.05`), early stopping on **val AUROC**.

---

## Experiment Matrix (key hyperparameters)

| ID | Model | img | P | D | L | H | MLP | Dropout | DropPath | Smooth | LR | WD | Epochs | Batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | ViT (scratch) | 224 | 16 | 256 | 6 | 8 | 4.0 | 0.1 | **0.05** | 0.05 | 3e‑4 | 0.05 | 15 | 32 |
| E2 | ViT‑B/16 (pretrained) | 224 | 16 | 768 | 12 | 12 | 4.0 | 0.0 | 0.0 | 0.05 | 1e‑4 | 0.05 | 10 | 32 |

> Monitoring: `va_auroc`, patience=5, **recall target** = 0.95 for @Rec.

---

## Results (final runs)

**Split:** `(4172, 1044, 624)`

### E1 — ViT (scratch) — ~5.0M params, ~28.4 min

- **VAL calibrated threshold**: `thr≈0.289` (VAL recall≈0.951, precision≈0.961)
- **TEST @0.5** → **acc 0.7933 · prec 0.7843 · rec 0.9231 · F1 0.8481 · AUROC 0.8892**  
- **TEST @Rec** → **acc 0.7788 · prec 0.7520 · rec 0.9641 · F1 0.8449 · AUROC 0.8892**

### E2 — ViT‑B/16 (pretrained) — ~85.8M params, ~45.4 min

- **VAL calibrated threshold**: `thr≈0.216` (VAL recall≈0.985, precision≈0.999)
- **TEST @0.5** → **acc 0.8702 · prec 0.8337 · rec 0.9897 · F1 0.9050 · AUROC 0.9007**  
- **TEST @Rec** → **acc 0.8413 · prec 0.7988 · rec 0.9974 · F1 0.8871 · AUROC 0.9007**

**Takeaways**
- Both models meet the **recall‑first** requirement when calibrated (@Rec).  
- **Pretrained** is stronger overall and converges faster, but heavier.  
- **Scratch** remains competitive on recall with much lower parameter count.

---

## Training Curves & Behavior

- With the tiny Kaggle `val/` (N=16), metrics and thresholds were **unstable**.  
  The **stratified 20% val** fixed this: smooth **val loss**, rising **val AUROC**, and stable calibration.
- **Scratch**: steady convergence; mild overfitting is mitigated by **DropPath** + smoothing.  
- **Pretrained**: rapid convergence, very high VAL metrics; small generalization gap to TEST.

---

## Issues Faced & Fixes

- **Kaggle download over corporate proxy** → Provided CLI/API fallback and **manual unzip** instructions.
- **PyTorch 2.6 `torch.load`** (weights_only default) → Added **safe unpickling** + resave **state_dict_only**.
- **Blocked timm/HuggingFace** → Used **offline Google `.npz`** and custom **JAX→PyTorch** weight mapper.
- **Threshold looked extreme (≈0.01)** with tiny VAL → after **larger stratified VAL**, threshold became **reasonable** (≈0.289 scratch, ≈0.216 pretrained).

---

## Strengths & Weaknesses

**ViT (scratch)**  
**+** Lightweight, good recall after calibration, simple to train.  
**–** Lower precision/AUROC than pretrained; benefits from more data/augmentations.

**ViT‑B/16 (pretrained)**  
**+** Best overall metrics & recall; fast convergence.  
**–** Heavy; needs offline weights under network restrictions.

---

## Suggested Next Steps

- Augmentation (RandAugment, stronger random crop, CutMix/Mixup), class‑balanced sampling.
- Losses: focal loss for imbalanced recall/precision trade‑off.
- Schedules: cosine LR with warmup, **layer‑wise LR decay** for pretrained.
- Resolution: 384×384 with pos‑emb interpolation.
- Calibration: temperature scaling on validation.
- Explainability: attention roll‑out, Grad‑CAM for clinical insight.

