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

