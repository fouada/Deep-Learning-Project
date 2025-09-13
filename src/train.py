from __future__ import annotations
import time, numpy as np, torch, torch.nn as nn
from .optimizers import build_optimizer, build_scheduler, SAM

# ---- AMP helpers (no warnings on CPU/MPS) ----
class _NoOpScaler:
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass
    def unscale_(self, opt): pass

def _make_scaler():
    if torch.cuda.is_available():
        try:
            return torch.amp.GradScaler('cuda')
        except Exception:
            return torch.cuda.amp.GradScaler()
    return _NoOpScaler()

def _autocast_enabled():
    return torch.cuda.is_available()

# ---- Early stopping ----
class EarlyStopper:
    def __init__(self, patience=10, mode='max'):
        self.patience = patience; self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.count = 0; self.should_stop = False
    def step(self, value):
        improved = (value > self.best) if self.mode == 'max' else (value < self.best)
        if improved:
            self.best = value; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience: self.should_stop = True
        return improved

# ---- Training loop ----
def train_model(model, train_loader, val_loader, device, *, arch: str,
                epochs: int, base_lr: float, weight_decay: float, label_smoothing: float,
                use_sam: bool, sam_rho: float | None,
                use_cosine: bool, warmup_epochs: int, min_lr_mult: float,
                monitor: str, patience: int, clip_grad_norm: float,
                class_weights=None, llrd=False, head_lr_mult=10.0, use_amp=False, tag: str = "model"):

    model = model.to(device)
    cw = class_weights.to(device) if class_weights is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=float(label_smoothing) if label_smoothing else 0.0)

    optimizer = build_optimizer(model, arch, base_lr, weight_decay,
                                use_sam=use_sam, sam_rho=(sam_rho or 0.05),
                                llrd=llrd, head_lr_mult=head_lr_mult)
    scheduler = build_scheduler(optimizer, use_cosine, epochs, warmup_epochs, min_lr_mult)
    early = EarlyStopper(patience=patience, mode='max' if monitor.startswith('va_') else 'min')

    scaler = _make_scaler()
    history = {'tr_loss': [], 'va_loss': [], 'va_acc': [], 'va_f1': [], 'va_auroc': []}
    best_state = None; best_epoch = -1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tr_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).long()

            if _autocast_enabled() and use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
            else:
                logits = model(xb); loss = loss_fn(logits, yb)

            if isinstance(optimizer, SAM):
                loss.backward()
                if clip_grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.first_step(zero_grad=True)

                if _autocast_enabled() and use_amp:
                    with torch.amp.autocast('cuda'):
                        logits2 = model(xb)
                        loss2 = loss_fn(logits2, yb)
                else:
                    loss2 = loss_fn(model(xb), yb)
                loss2.backward()
                if clip_grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.second_step(zero_grad=True)
            else:
                scaler.scale(loss).backward()
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_losses.append(float(loss.detach().cpu().item()))

        if scheduler is not None: scheduler.step()

        # --- Validation ---
        model.eval(); va_losses = []; correct = 0; n = 0
        probs = []; targs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).long()
                logits = model(xb); loss = loss_fn(logits, yb)
                va_losses.append(float(loss.detach().cpu().item()))
                p = torch.softmax(logits, dim=1)[:, 1]
                probs.append(p.detach().cpu().numpy())
                targs.append(yb.detach().cpu().numpy())
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item(); n += yb.numel()

        from sklearn.metrics import f1_score, roc_auc_score
        probs = np.concatenate(probs); targs = np.concatenate(targs)
        va_acc = correct / max(1, n)
        va_f1  = f1_score(targs, (probs >= 0.5).astype(int), zero_division=0)
        try:
            va_auroc = roc_auc_score(targs, probs)
        except Exception:
            va_auroc = float('nan')

        history['tr_loss'].append(float(np.mean(tr_losses)))
        history['va_loss'].append(float(np.mean(va_losses)))
        history['va_acc'].append(float(va_acc))
        history['va_f1'].append(float(va_f1))
        history['va_auroc'].append(float(va_auroc))

        metric = {'va_acc': va_acc, 'va_f1': va_f1, 'va_auroc': va_auroc}.get(monitor, va_auroc)
        improved = early.step(metric)
        print(f"[{tag}] epoch {ep:02d}/{epochs:02d}  tr_loss={np.mean(tr_losses):.4f}  "
              f"val_loss={np.mean(va_losses):.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}  "
              f"val_auroc={va_auroc:.4f}  (no_improve={early.count})")

        if improved:
            best_state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}
            best_epoch = ep

        if early.should_stop:
            print(f"[{tag}] early stopping at epoch {ep}.")
            break

    if best_state is not None: model.load_state_dict(best_state)
    return model, history, (time.time() - t0)
