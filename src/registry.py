from __future__ import annotations
import numpy as np, torch

result_registry = {}  # name -> dict

def _sum_params(model):
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None

def add_result(name: str, model, summary_05: dict, summary_recall: dict,
               threshold=None, time_sec=None):
    result_registry[name] = {
        "summary@0.5": summary_05,
        "summary@recall": summary_recall,
        "threshold": float(threshold) if threshold is not None else None,
        "time_sec": float(time_sec) if time_sec is not None else None,
        "params": _sum_params(model),
    }
    return result_registry[name]
