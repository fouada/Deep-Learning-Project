from __future__ import annotations
import math, inspect, torch
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

# ----- SAM wrapper -----
class SAM(Optimizer):
    """Sharpness-Aware Minimization (wrapper)."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        # filter kwargs to those supported by base optimizer
        sig = inspect.signature(base_optimizer.__init__)
        valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
        defaults = dict(rho=rho, **valid)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **valid)
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale)
        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError("Use first_step/second_step with SAM")

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                norms.append(torch.norm(p.grad.detach(), p=2).to(device))
        if not norms:
            return torch.tensor(0., device=device)
        return torch.norm(torch.stack(norms), p=2)

# ----- Cosine schedule with warmup -----
def cosine_with_warmup(optimizer, num_epochs: int, warmup_epochs: int, min_lr_mult: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        t = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return min_lr_mult + 0.5 * (1.0 - min_lr_mult) * (1.0 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# ----- Builders -----
def build_optimizer(model, arch: str, base_lr: float, weight_decay: float,
                    use_sam: bool, sam_rho: float, llrd: bool, head_lr_mult: float):
    """
    arch: 'resnet18', 'vit', 'custom_cnn'
    - ResNet/CustomCNN scratch: SGD momentum 0.9 + WD ~1e-3
    - ViT scratch: AdamW + WD ~0.3
    - Pretrained: same optimizer, smaller LR; LLRD supported for ViT (head gets higher LR)
    """
    params = model.parameters()
    if arch in ["resnet18", "custom_cnn"]:
        # default SGD for CNNs
        base_opt = SGD
        lr = base_lr if base_lr is not None else 0.1
        wd = weight_decay if weight_decay is not None else 1e-3
        kwargs = dict(lr=lr, momentum=0.9, weight_decay=wd)
        return SAM(params, base_opt, rho=sam_rho, **kwargs) if use_sam else base_opt(params, **kwargs)

    # ViT
    base_opt = AdamW
    lr = base_lr if base_lr is not None else 3e-4
    wd = weight_decay if weight_decay is not None else 0.3

    if llrd:
        # Head gets larger LR
        head = []
        body = []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            (head if n.startswith("head") else body).append(p)
        opt_params = [
            {"params": body, "lr": lr, "weight_decay": wd},
            {"params": head, "lr": lr * head_lr_mult, "weight_decay": wd},
        ]
    else:
        opt_params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": wd}]

    return SAM(opt_params, base_opt, rho=sam_rho) if use_sam else base_opt(opt_params, lr=lr, weight_decay=wd)

def build_scheduler(optimizer_or_sam, use_cosine: bool, num_epochs: int, warmup_epochs: int, min_lr_mult: float):
    base_opt = optimizer_or_sam.base_optimizer if isinstance(optimizer_or_sam, SAM) else optimizer_or_sam
    if not use_cosine: return None
    return cosine_with_warmup(base_opt, num_epochs, warmup_epochs, min_lr_mult)
