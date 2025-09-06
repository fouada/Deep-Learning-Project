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
