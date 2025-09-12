from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@torch.no_grad()
def evaluate_probs_targets(model, loader, device):
    model.eval(); probs=[]; targs=[]
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        p = F.softmax(out, dim=1)[:,1].detach().cpu().numpy()
        probs.append(p)
        targs.append(yb.numpy())
    probs = np.concatenate(probs); targs = np.concatenate(targs)
    try:
        auroc = roc_auc_score(targs, probs)
    except Exception:
        auroc = float('nan')
    return probs, targs, auroc

def evaluate(model, loader, device):
    probs, targs, auroc = evaluate_probs_targets(model, loader, device)
    return {'targets': targs, 'probs': np.stack([1-probs, probs], axis=1), 'auroc': auroc}

def choose_threshold_by_min_recall(y_true, y_prob, target_recall=0.95):
    # grid search over unique scores
    order = np.argsort(y_prob)
    sorted_scores = y_prob[order]
    best_thr = 0.5; best_prec=0.0; best_rec=0.0; best_f1=0.0
    for thr in np.unique(sorted_scores):
        yhat = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, yhat, zero_division=0)
        rec  = recall_score(y_true, yhat, zero_division=0)
        f1   = f1_score(y_true, yhat, zero_division=0)
        if rec >= target_recall:
            best_thr, best_prec, best_rec, best_f1 = thr, prec, rec, f1
            break
    return float(best_thr), float(best_rec), float(best_prec), float(best_f1)

def summarize_at_threshold(y_true, y_prob, thr):
    yhat = (y_prob >= thr).astype(int)
    return {
        "acc": accuracy_score(y_true, yhat),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "f1": f1_score(y_true, yhat, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "thr": float(thr),
    }
