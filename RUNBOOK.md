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
