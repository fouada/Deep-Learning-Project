
# Vision Transformer (ViT) on Chest X‑ray — Results vs. ViT Paper (ICLR’21)  

**Project**: Pneumonia (binary) classification on Chest X‑ray  
**Notebook**: `ViT_ChestXRay_Binary_Classification_FINAL_1.ipynb` (latest working runs)  
**Dataset split used for the final runs**: **Train 4,172 / Val 1,044 / Test 624** images (stratified).  
**Classes**: `NORMAL` (neg), `PNEUMONIA` (pos).  
**Metric focus**: **Recall-first** (screening) – high sensitivity for PNEUMONIA, then maximize precision.

---

## 1) What the ViT paper says (condensed)

- **ViT has weaker image inductive biases than CNNs** (no built‑in translation equivariance / locality), so **training *from scratch* on small data underperforms CNNs**; **pre‑training at scale changes the picture** (“large‑scale training trumps inductive bias”).  
- **Pre‑train at scale** (ImageNet‑21k / JFT‑300M), then **fine‑tune** on the downstream task; **fine‑tune at higher resolution** with **2‑D interpolation of positional embeddings** if needed.  
- **Scaling laws**: Larger ViT models shine with more data/compute; hybrids (CNN backbone + ViT) may help at small budgets; depth gives strong gains; ViT can be compute‑efficient at scale.

_Implications for medical imaging_: Without large pre‑training, ViT may struggle; with pre‑training, it transfers strongly even on specialized domains.

---

## 2) Your setup (what we trained & how)

### Models
- **Custom ViT (scratch)**: `img=224, P=16, D=256, L=6, heads=8, MLP ratio=4, drop=0.1, drop‑path=0.05`  
  ~5.0M params. Designed deliberately **smaller than ViT‑B/16** to match data/compute.
- **Pretrained ViT‑B/16**: Google **JAX .npz** weights (D=768, L=12, heads=12; ~85.8M params), head replaced by 2‑class linear layer; **pos‑emb interpolation** handled if grid differs.

### Training
- **Optimizer / regularization**: AdamW, weight decay (0.05), label smoothing (0.05), dropout, stochastic depth (drop‑path).  
- **Recall‑first calibration**: Pick the **lowest threshold** on **VAL** that achieves **target recall** (we used ≈0.95), and among those thresholds **maximize precision**; evaluate both **@0.5** and **@Rec** on TEST.
- **Loaders**: Same Kaggle chest X‑ray pipeline for both models; transforms convert grayscale→RGB.

---

## 3) Quantitative results (TEST)

### 3.1  Scratch ViT (6‑layer, ~5.0M params)

| Threshold | Acc | Precision | Recall | F1 | AUROC | Chosen thr |
|---|---:|---:|---:|---:|---:|---:|
| **@0.5** | 0.7933 | 0.7843 | 0.9231 | 0.8481 | 0.8892 | 0.5000 |
| **@Rec** | 0.7788 | 0.7520 | **0.9641** | 0.8449 | 0.8892 | **0.2890** |

**Observation.** The recall‑first operating point substantially increases sensitivity (0.964↑) with a tolerable precision trade‑off, as expected for screening.

### 3.2  Pretrained ViT‑B/16 (~85.8M params)

| Threshold | Acc | Precision | Recall | F1 | AUROC | Chosen thr |
|---|---:|---:|---:|---:|---:|---:|
| **@0.5** | 0.8702 | 0.8337 | **0.9897** | 0.9050 | 0.9007 | 0.5000 |
| **@Rec** | 0.8413 | 0.7988 | **0.9974** | 0.8871 | 0.9007 | **0.2160** |

**Observation.** With recall‑first calibration, **recall ≈ 1.0** while **precision** remains strong (>0.79). This meets a screening goal with fewer false negatives than the scratch model.

### 3.3  Complexity & wall‑clock

| Model | Params (M) | Train time (min) | Test AUROC (@Rec) |
|---|---:|---:|---:|
| **ViT (scratch)** | **4.99** | **28.39** | **0.8892** |
| **ViT‑B/16 (pretrained)** | **85.80** | **45.36** | **0.9007** |

**Trade‑off.** Pretrained ViT costs more time/params but **converges faster** to high recall/precision and better AUROC.

---

## 4) Training curves & generalization

- **Scratch ViT**: Training/validation loss **monotonic down**; validation **F1 ≈ 0.94** by epoch 15; no severe overfitting with current regularization (label smoothing, dropout, drop‑path).  
- **Pretrained ViT**: Very **rapid convergence** (≤10 epochs), consistently high validation F1 (≥0.97), and stable AUROC.  
- **Effect of validation size**: Moving from a tiny VAL (N=16) to a **stratified 20% VAL (N=1,044)** produced **stable threshold estimates** and much more reliable recall‑first calibration.

---

## 5) How these results align with the ViT paper

| Paper claim | What we see here |
|---|---|
| **ViT needs scale**: Weak CNN inductive biases (translation-equivariance, locality) → ViT from scratch on small data underperforms; **pre‑training at scale flips the result**. | **Scratch ViT** is solid but lags the **pretrained ViT‑B/16**, especially at **recall‑first** operating points. This directly mirrors the paper’s finding that **pre‑training trumps inductive bias** when transferring to smaller tasks/domains. |
| **Fine‑tune at higher resolution; interpolate positional embeddings** when grid differs. | Our loader supports **pos‑emb interpolation**; with chest X‑rays at 224 and patch‑size 16, grids align; the code is robust to resolution changes. |
| **Hybrids can help at small budgets**; **depth scales well**. | Our **small scratch ViT (6‑layer)** is appropriate for available data/compute. If we need more performance **without very large pre‑training**, a **CNN+ViT hybrid** is a good next step. |

---

## 6) Strengths & weaknesses in this task

**Strengths**
- **Pretrained ViT‑B/16** hits **recall ≈ 1.0** with **precision ≈ 0.80** — **screening‑grade sensitivity** with manageable false positives.  
- **AUROC ≥ 0.90** indicates robust ranking quality for both models; PR curves in the notebook show good precision over a wide recall range.

**Weaknesses / risks**
- **Data hunger**: Scratch ViT needs either **more data** or **stronger inductive bias**; performance degrades with very small VAL/TEST.  
- **Compute**: Pretrained ViT has ~**86M params**; memory/latency can be a concern on limited hardware.

---

## 7) Recommendations

1. **Keep recall‑first calibration** in deployment (threshold from a stratified/representative VAL).  
2. **Lightweight hybrid** for scratch path: CNN stem → ViT encoder (adds locality) to improve small‑data generalization.  
3. **Augmentations** (medical‑safe): flips, small rotations, CLAHE, MixUp/CutMix (tuned).  
4. **Freeze early blocks** of the pretrained model for stability; unfreeze progressively if more data arrives.  
5. **Monitor per‑class errors** (NORMAL→PNEUMONIA vs PNEUMONIA→NORMAL) to quantify clinical impact.

---

## 8) Take‑home messages

- Your experiments **validate the ViT paper’s core thesis**: **pre‑training at scale** makes ViT **excel on smaller/specialized datasets**, delivering **near‑perfect recall** with strong precision when calibrated.  
- With the **recall‑first** operating point, both models meet a **screening** requirement; the **pretrained ViT** does so with **better precision/accuracy** and slightly higher AUROC.

