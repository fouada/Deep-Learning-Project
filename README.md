# Deep-Learning-Project



# Chest X‑ray Pneumonia Classification — Vision Transformer (ViT)

This repository contains a **from‑scratch Vision Transformer** (ViT) implementation and a **pretrained ViT‑B/16** fine‑tuning path for binary classification (*PNEUMONIA* vs *NORMAL*) on the **Kaggle Chest X‑ray dataset**.  
The project is designed for a **recall‑first** screening use‑case and mirrors the evaluation decisions taken in an earlier CNN notebook, while keeping the ViT implementation faithful to the original paper.

> **Key notebooks**  
>  — final, working notebook with clean cell order:  
>   1) Utilities → 2) Data (stratified split) → 3) Model (scratch ViT + DropPath) → 4) Train/Eval (scratch) →  
>   5) Pretrained (Google .npz) → 6) Train/Eval (pretrained) → 7) Comparison & plots
>
> **Final split (stratified from the original training set):**  
> `len(train_ds), len(val_ds), len(test_ds) = (4172, 1044, 624)`

---

## 1) Goals

1. Implement a **CNN baseline** (reference; not shared here) and a **ViT** to solve the same binary task.  
2. Implement **two ViT variants**:  
   - **Custom ViT from scratch** (patchify, [CLS] token, 1‑D pos‑emb, pre‑LN, GELU MLP, **DropPath**).  
   - **Pretrained ViT‑B/16** using Google’s **JAX `.npz`** checkpoint (offline) loaded into our PyTorch ViT skeleton.  
3. Evaluate with a **recall‑first operating point**, then compare **Accuracy, Precision, Recall, F1, AUROC**, plus training curves and complexity.
4. Make the notebook robust in corporate networks (proxies, SSL) and across PyTorch versions (safe checkpoint loading).

---

## 2) Environment

A minimal conda environment (CPU or Apple MPS will work; CUDA preferred if available):

```bash
conda create -n vitfinalproject python=3.11 -y
conda activate vitfinalproject

# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # pick CUDA build if you have it
pip install numpy matplotlib scikit-learn
# Optional (only if you want to try timm online; the project uses offline .npz weights instead)
pip install timm

# If you plan to use Kaggle CLI:
pip install kaggle
```

> On Apple MPS you may see: **“pin_memory is set but not supported on MPS”** — harmless, training proceeds normally.

---

## 3) Data

Dataset: **Chest X‑Ray Images (Pneumonia)** (`paultimothymooney/chest-xray`)  
Expected layout: `Data/chest_xray/{train,val,test}/...`

You can:  
- **Manual**: download the zip from Kaggle, unzip so the folder above exists.  
- **CLI** (if network allows):
  ```bash
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p Data --unzip
  ```

> The repo keeps `Data/` **out of Git**. The `.gitignore` contains:
> ```
> .DS_Store
> Data/*
> !Data/.gitkeep
> ```

A **stratified validation split (20%)** is created **from the original training set** to avoid the tiny Kaggle `val/` (which has only 16 images). This stabilizes validation metrics and threshold calibration.

---

## 4) Pretrained weights (offline, no `timm` needed)

Download **Google’s JAX ViT‑B/16** checkpoint and place it here:

```
notebooks/weights/ViT-B_16.npz
```

Example (outside corporate proxy):
```bash
mkdir -p notebooks/weights
curl -L -o notebooks/weights/ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

The notebook includes a **JAX→PyTorch loader** that maps the `.npz` arrays into our PyTorch ViT‑B/16 (patch embedding, q/k/v merge, out‑proj, MLPs, LayerNorms, pos‑emb interpolation). We keep our **2‑class head** and fine‑tune.

---

## 5) Models

### 5.1 Custom ViT (from scratch)
- `img_size=224`, `patch_size=16`, `[CLS]` token + learnable 1‑D positional embeddings
- `embed_dim=256`, `depth=6`, `num_heads=8`, `mlp_ratio=4`
- **Pre‑norm** Transformer blocks (LN → MSA → residual; LN → MLP → residual)
- **Dropout** + **DropPath** (stochastic depth) with a small global rate (e.g., `0.05`)
- **Label smoothing** `0.05`, **AdamW** (`lr=3e-4`, `wd=0.05`), **early stopping** on **val AUROC**

> **DropPath insertion** (inside the Transformer block, applied to each residual branch):
> ```python
> class DropPath(nn.Module):
>     def __init__(self, drop_prob=0.0):
>         super().__init__()
>         self.drop_prob = float(drop_prob)
>     def forward(self, x):
>         if self.drop_prob == 0.0 or not self.training:
>             return x
>         keep_prob = 1.0 - self.drop_prob
>         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
>         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
>         random_tensor.floor_()
>         return x / keep_prob * random_tensor
> ```
> Use it after attention and MLP: `x = x + drop_path(attn(norm1(x)))`, `x = x + drop_path(mlp(norm2(x)))` with a depth‑wise schedule.

### 5.2 Pretrained ViT‑B/16
- PyTorch skeleton with `embed_dim=768`, `depth=12`, `num_heads=12` (head = 2 classes)
- Load **`ViT-B_16.npz`** offline; fine‑tune with AdamW (`lr=1e-4`, `wd=0.05`), same criterion and early stopping

---

## 6) Training & Evaluation

### 6.1 Loss and regularization
- Weighted **CrossEntropy** with optional **class weights**
- **Label smoothing** (`0.05`)
- **DropPath** for the scratch model
- **Early stopping** (patience=5) monitoring **val AUROC**

### 6.2 Recall‑first evaluation
For screening, we want **very high recall**. We **calibrate on validation** by scanning thresholds and choosing the **highest‑precision threshold** that achieves **recall ≥ target** (default **0.95**). That threshold is then used on **TEST**.

The notebook provides:
```python
def choose_threshold_by_min_recall(y_true, p1, min_recall=0.95):
    # returns (best_thr, val_recall_at_thr, val_precision_at_thr)
    ...
def summarize_at_threshold(y_true, p1, thr):
    # returns dict: acc, precision, recall, f1, auroc, thr
    ...
```

We report metrics at **two operating points**:
- **@0.5** — fixed threshold 0.5  
- **@Rec** — recall‑first calibrated threshold from VAL

---

## 7) Results (final runs)

**Split:** `(train, val, test) = (4172, 1044, 624)`

### 7.1 Scratch ViT (≈5.0M params, ≈28.4 min)

- **VAL‑calibrated threshold:** `thr≈0.289` (VAL recall≈0.951, precision≈0.961)
- **TEST @0.5:**  
  **acc 0.7933** · **prec 0.7843** · **rec 0.9231** · **F1 0.8481** · **AUROC 0.8892**
- **TEST @Rec (thr≈0.289):**  
  **acc 0.7788** · **prec 0.7520** · **rec 0.9641** · **F1 0.8449** · **AUROC 0.8892**

### 7.2 Pretrained ViT‑B/16 (≈85.8M params, ≈45.4 min)

- **VAL‑calibrated threshold:** `thr≈0.216` (VAL recall≈0.985, precision≈0.999)
- **TEST @0.5:**  
  **acc 0.8702** · **prec 0.8337** · **rec 0.9897** · **F1 0.9050** · **AUROC 0.9007**
- **TEST @Rec (thr≈0.216):**  
  **acc 0.8413** · **prec 0.7988** · **rec 0.9974** · **F1 0.8871** · **AUROC 0.9007**

### 7.3 Comparison table (recall‑first)

| Model | Params (M) | Time (min) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch)** | 4.99 | 28.39 | 0.7788 | 0.7520 | **0.9641** | 0.8449 | 0.8892 | 0.289 |
| **ViT (pretrained)** | 85.80 | 45.36 | 0.8413 | 0.7988 | **0.9974** | 0.8871 | 0.9007 | 0.216 |

**Observations**
- Both models achieve **very high recall** under recall‑first calibration (goal achieved).  
- **Pretrained** consistently delivers better overall metrics and converges faster.  
- **Scratch** is lighter and still competitive after calibration, but with lower precision/AUROC.

---

## 8) Training curves & generalization

- The original Kaggle `val/` (N=16) made metrics unstable. With **stratified 20% val**, curves are stable:  
  - **Scratch** shows steady improvement; AUROC ≈ **0.89** on TEST.  
  - **Pretrained** converges quickly; AUROC ≈ **0.90** on TEST.  
- **Recall‑first** shifts the operating point: **recall ↑**, **precision/accuracy ↓** mildly; **AUROC** unchanged (threshold‑independent).

---

## 9) Complexity

| Model | Params | Notes |
|---|---:|---|
| ViT (scratch) | ~**5.0M** | light, faster to train |
| ViT‑B/16 | ~**85.8M** | heavy, best metrics; requires offline `.npz` in locked networks |

---

## 10) Strengths & weaknesses

**ViT (scratch)**  
**+** Small, simple, solid recall after calibration.  
**–** Lower precision and AUROC compared to pretrained; ceiling limited without more data/regularization.

**ViT‑B/16 (pretrained)**  
**+** Excellent recall and strong overall metrics; fast convergence.  
**–** Heavy; needs offline weight management in restricted networks.

---

## 11) Troubleshooting & fixes (what we learned)

- **Kaggle download**: SSL/DNS issues are common behind proxies. The notebook offers CLI → API → manual fallback and a clear message on where to place data.  
- **Checkpoints on PyTorch ≥ 2.6**: use **safe unpickling** or save **`state_dict_only.pth`**. The notebook includes a safe loader and resaves a clean state dict.  
- **HuggingFace/timm blocked**: we avoid online fetch by using **Google’s `.npz`** and a custom loader.  
- **Validation size matters**: the tiny Kaggle `val/` produced unstable thresholds; **stratified 20%** fixed calibration.  
- **Recall‑first** is a **thresholding** problem; do it on **validation** and report both **@0.5** and **@Rec**.

---

## 13) Next steps (optional)

- Stronger data augmentation: RandAugment / RandomResizedCrop / mild ColorJitter; Mixup/CutMix.  
- Class‑balanced sampler or **focal loss** to trade precision/recall more smoothly.  
- **Layer‑wise LR decay** for pretrained; **cosine LR + warmup**; gradient clipping.  
- **Higher resolution** (e.g., 384×384) with pos‑emb interpolation.  
- Probability calibration (temperature scaling) and **explainability** (attention rollout / Grad‑CAM).

---

## 14) Acknowledgement

- ViT architecture inspired by *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”* (Dosovitskiy et al., 2020).  
- Pretrained **ViT‑B/16** checkpoint courtesy of Google Research.




# ANALYSIS.md — Executive Summary (ViT vs CNN Baseline Context)

**Task**: Binary classification of chest X‑rays (*NORMAL* vs *PNEUMONIA*) on the Kaggle dataset.  
**Goal**: Prioritize **recall** (sensitivity) for screening—catch as many pneumonia cases as possible—even if it slightly reduces precision.

**Pipelines compared**
- **ViT (scratch)** — custom implementation faithful to the paper (patch‑linear‑embed → [CLS]+pos → L×{Pre‑LN → MSA → residual → Pre‑LN → MLP(GELU) → residual} → LN → [CLS] → linear head), with **DropPath** and **label smoothing**.
- **ViT‑B/16 (pretrained)** — initialized offline from Google **JAX `.npz`** weights (`ViT-B_16.npz`), mapped into an equivalent PyTorch skeleton, then fine‑tuned.

Both use **stratified 20% validation** (from Kaggle train) to stabilize calibration and early stopping on **val AUROC**. Test set is Kaggle’s official **test/**.

---

## Key Findings

- **High recall** is achievable on both models with **VAL‑calibrated thresholds** (recall‑first).  
- **Pretrained ViT‑B/16** outperforms scratch in almost every metric and **converges faster**, at the cost of **~17× more parameters** and longer training time.  
- The initial instability seen with the tiny Kaggle `val/` (16 images) was resolved by **stratified splitting**; the learned thresholds became **reasonable** instead of extreme.

---

## Final Metrics (TEST set)

**Split used**: `(train, val, test) = (4172, 1044, 624)`; classes `['NORMAL', 'PNEUMONIA']`

### Operating points
- **@0.5** — fixed threshold = 0.5 (baseline view).  
- **@Rec** — **recall‑first**: choose the **highest‑precision** threshold on **VAL** s.t. recall ≥ **0.95**; apply it to **TEST**.

### Summary table (@Rec emphasized for screening)

| Model | Params (M) | Time (min) | Acc | Prec | **Recall** | F1 | AUROC | Thr (@Rec) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch)** | **4.99** | 28.39 | 0.7788 | 0.7520 | **0.9641** | 0.8449 | 0.8892 | 0.289 |
| **ViT‑B/16 (pretrained)** | **85.80** | 45.36 | 0.8413 | 0.7988 | **0.9974** | 0.8871 | 0.9007 | 0.216 |

For context (@0.5):
- **Scratch**: acc 0.7933 · prec 0.7843 · rec 0.9231 · F1 0.8481 · AUROC 0.8892  
- **Pretrained**: acc 0.8702 · prec 0.8337 · rec 0.9897 · F1 0.9050 · AUROC 0.9007

**Interpretation**: Under the recall‑first objective, **Pretrained ViT‑B/16** is the best choice; **Scratch ViT** is competitive for recall with far lower complexity.

---

## Training Curves, Convergence, and Generalization

- With the tiny built‑in `val/` (16 images), validation curves and thresholds were noisy/unreliable.  
  After switching to a **stratified 20% validation split**, curves became **smooth**, **val AUROC** rose consistently, and **threshold calibration stabilized**.
- **ViT (scratch)**: steady loss drop; minor overfitting mitigated by **DropPath** and **label smoothing**; AUROC stable around ~0.89 on TEST.  
- **ViT‑B/16 (pretrained)**: very fast convergence; **val** metrics are near‑ceiling, with a small but expected generalization gap on **TEST**, likely due to domain shift.

---

## Model Complexity

- **ViT (scratch)**: ~**5.0M** params; ~**28.4** minutes (MPS).  
- **ViT‑B/16 (pretrained)**: ~**85.8M** params; ~**45.4** minutes (MPS).
- Memory: both fit at **224×224** and batch size 32 on typical MPS/CPU; larger resolution requires more memory.

---

## Observations: Over/Under‑fitting

- The pretrained model, without heavy augmentation, can **over‑fit VAL** if VAL is too small; the stratified split cured most volatility.  
- The scratch model benefits from **regularization** (DropPath, smoothing) and enough **epochs**; performance improves with consistent validation and thresholding.

---

## Strengths & Weaknesses (ViT only)

**ViT (scratch)**  
**✔** Light (~5M), good recall after calibration, reproducible.  
**✖** Lower AUROC/precision than pretrained; benefits from more data and augmentation.

**ViT‑B/16 (pretrained)**  
**✔** Best metrics overall; near‑perfect recall under @Rec; fastest convergence.  
**✖** Heavy (~86M), longer training; needs offline weights under restricted networks.

---

## Techniques Applied / Alternatives to Try

- **Already used**: label smoothing (0.05), **DropPath** (0.05 linear schedule), class weights, early stopping on **val AUROC**, **recall‑first thresholding**.
- **Recommended next**:  
  - **Augmentations**: RandAugment / stronger random crop; **Mixup/CutMix**.  
  - **Loss**: focal loss for precision‑recall balance on imbalanced data.  
  - **Optimization**: cosine LR with warmup; **layer‑wise LR decay** (pretrained).  
  - **Resolution**: 384×384 with position‑embedding interpolation.  
  - **Calibration**: temperature scaling on VAL to improve calibrated probabilities.  
  - **Explainability**: attention roll‑out / Grad‑CAM to visualize pulmonary regions.

---

## Repro Tips

 
  Run top‑to‑bottom: **scratch** (train→eval) → **pretrained** (train→eval) → **comparison**.
- Data layout: `Data/chest_xray/{train,val,test}/...` (keep `Data/` out of Git; use `Data/.gitkeep`).
- Pretrained weights: `notebooks/weights/ViT-B_16.npz` (offline).  
- Use **RUNBOOK.md** for exact instructions; **EXPERIMENTS.md** for full details.

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

## Results & Analysis (Executive Summary)

**Goal:** screening‑oriented **recall‑first** classification of chest X‑rays (*NORMAL* vs *PNEUMONIA*).  
**Models:** custom **ViT (scratch)** and **pretrained ViT‑B/16** (offline Google `.npz`).  
**Validation:** stratified 20% from train for stable calibration; early stopping on **val AUROC**.

### TEST metrics (two operating points)

- **@0.5** — fixed threshold = 0.5 (baseline)  
- **@Rec** — threshold from **VAL** that achieves **recall ≥ 0.95** with maximum precision; applied to **TEST**

| Model | Params (M) | Time (min) | Acc | Prec | **Recall** | F1 | AUROC | Thr (@Rec) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch)** | **4.99** | 28.39 | 0.7788 | 0.7520 | **0.9641** | 0.8449 | 0.8892 | 0.289 |
| **ViT‑B/16 (pretrained)** | **85.80** | 45.36 | 0.8413 | 0.7988 | **0.9974** | 0.8871 | 0.9007 | 0.216 |

**Summary**  
- Both models meet the screening objective under **@Rec**.  
- **Pretrained ViT‑B/16** leads overall and converges faster; **Scratch ViT** is competitive on recall with ~17× fewer parameters.

> See **EXPERIMENTS.md** for full context and **RUNBOOK.md** for reproduction steps.

# RUNBOOK.md — How to Run the ViT Experiments

This runbook gives **step‑by‑step instructions** to reproduce the **scratch ViT** and **pretrained ViT‑B/16** experiments with a **recall‑first** evaluation.

---

## 1) Setup

### 1.1 Create environment
```bash
conda create -n vitfinalproject python=3.11 -y
conda activate vitfinalproject

# Core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # use CUDA wheel if available
pip install numpy matplotlib scikit-learn

# Optional (only if you plan to try online timm; the project uses offline .npz)
pip install timm

# If you want Kaggle CLI convenience:
pip install kaggle
```

> On **Apple MPS** you may see:  
> `UserWarning: 'pin_memory' argument is set as true but not supported on MPS` — harmless.

### 1.2 Clone & open
```bash
git clone <your-repo-url> Deep-Learning-Project
cd Deep-Learning-Project
```

Launch Jupyter and open the final notebook:
```

```

---

## 2) Data

### 2.1 Layout
Expected path:
```
Data/chest_xray/{train,val,test}/...
```

### 2.2 Get the dataset
- **Manual (recommended for corporate networks):** download the Kaggle zip in a browser, unzip into `Data/` so the folder structure above exists.
- **CLI (if network allows):**
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p Data --unzip
```

### 2.3 Keep data out of Git
`.gitignore`:
```
.DS_Store
Data/*
!Data/.gitkeep
```

---

## 3) Pretrained weights (offline)

Download **Google ViT‑B/16** checkpoint and place it here:
```
notebooks/weights/ViT-B_16.npz
```

Example:
```bash
mkdir -p notebooks/weights
curl -L -o notebooks/weights/ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

> No HuggingFace/timm downloads needed; the notebook includes a **JAX→PyTorch** loader that maps this `.npz` into the PyTorch ViT‑B/16 skeleton.

---

## 4) Run Order (one click per section)

The notebook is organized so you can **run top‑to‑bottom**:

1) **Utilities & Config**  
   - Prints device (CPU/CUDA/MPS), resolves dataset/weights directories.  
   - Defines **DropPath**, training loop (`train_model`), evaluation (`evaluate`), plotting, and threshold helpers:  
     `choose_threshold_by_min_recall`, `summarize_at_threshold`, `plot_curve`.

2) **Data & Stratified Split**  
   - Creates a **20% validation split** from original train for stability.  
   - Prints **class counts** and dataset lengths `(train, val, test)`.

3) **Model A — Custom ViT (scratch)**  
   - Builds the ViT as in the paper (patchify → [CLS] + pos → L×{Pre‑LN + MSA + residual; Pre‑LN + MLP(GELU) + residual} → LN → [CLS] → head).  
   - **DropPath** is applied to both residual branches (attention & MLP) with a linear depth‑wise schedule.  
   - **Train** with AdamW, label smoothing, optional class weights, early stopping on **val AUROC**.  
   - **Evaluate**:  
     - Get **VAL probs**, run `choose_threshold_by_min_recall(..., min_recall=cfg.target_recall)`  
     - Report **TEST @0.5** and **TEST @Rec** (recall‑first) via `summarize_at_threshold`  
     - Show confusion matrix, ROC, PR if desired (helper cell provided).

4) **Model B — Pretrained ViT‑B/16 (.npz)**  
   - Instantiates ViT‑B/16 skeleton; loads **`ViT-B_16.npz`** (patch/pos/blocks/LN mapping).  
   - **Fine‑tune** with AdamW; same training loop, early stopping on **val AUROC**.  
   - **Evaluate** like scratch (**VAL calibration** → **TEST @0.5** and **@Rec**).

5) **Comparison Table & Plots**  
   - Consolidates both models into a **single table** (params, time, Acc/Prec/Recall/F1/AUROC, threshold).  
   - Optional diagnostic plots for both (confusion, ROC, PR).

---

## 5) Changing the Recall Target

- In the notebook, set:
```python
cfg.target_recall = 0.95   # or 0.98 for more aggressive screening
```
(or change the `TARGET_RECALL` constant in the evaluation cell).  
The calibration step will search for the **highest precision** threshold that still achieves the target **recall** on **VAL**. That threshold is then used on **TEST**.

---

## 6) Expected Final Numbers (reference)

**Split:** `(4172, 1044, 624)`

- **ViT (scratch)** @Rec (`thr≈0.289`): **acc 0.7788 · prec 0.7520 · rec 0.9641 · F1 0.8449 · AUROC 0.8892**  
- **ViT‑B/16 (pretrained)** @Rec (`thr≈0.216`): **acc 0.8413 · prec 0.7988 · rec 0.9974 · F1 0.8871 · AUROC 0.9007**

(Also reported @0.5 in the notebook for context.)

---

## 7) Troubleshooting

- **Kaggle download fails (SSL/DNS/proxy)** → Use **manual** download & unzip into `Data/`. The notebook prints exactly where it expects the data.
- **`torch.load` fails on PyTorch ≥ 2.6** → Use the **safe loader** cell (already in notebook) or load weights‑only `.pth` files. The notebook resaves a clean `*_state_dict_only.pth` automatically.
- **HuggingFace/timm blocked** → We do not need online timm weights. Use the offline `.npz` in `notebooks/weights/`.
- **MPS warning** about `pin_memory` → ignore; set `num_workers=0` on macOS for stability.
- **Validation too small / unstable** → ensure you are using the **stratified 20% split**, not the tiny Kaggle `val/` folder.

---

## 8) Tips

- If recall is **too low** even after calibration, check **class weights**, **augmentations**, or raise `cfg.target_recall` to push the threshold lower (recall ↑, precision ↓).  
- If precision is too low, consider **focal loss**, **class‑balanced sampler**, or relax the recall target slightly.
- Try **cosine LR + warmup** and **layer‑wise LR decay** (especially for the pretrained model).
- For better AUROC, add augmentations or increase input **resolution** to 384×384 (the notebook’s loader will interpolate pos‑emb).

---

## 9) Re‑exporting results

At the end, the notebook collects a Pandas DataFrame for comparison. Save it as:
```python
df.to_csv("notebooks/outputs/vit_comparison.csv", index=False)
```

This file can be committed and used in your report.



