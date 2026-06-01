"""Validation-based detector for label-flip style updates.

Weight fingerprints catch obvious parameter attacks, but label flip often keeps
the weights statistically plausible. This checker scores each uploaded model on
a clean public validation loader and rejects round outliers whose loss is worse
than the current global model.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class PublicValidationCheck:
    def __init__(
        self,
        val_loader,
        device: str | torch.device,
        min_delta: float = 0.02,
        mad_k: float = 3.0,
    ) -> None:
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.min_delta = float(min_delta)
        self.mad_k = float(mad_k)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    @torch.no_grad()
    def _loss(self, model: torch.nn.Module) -> float:
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total = 0
        for x, y in self.val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            out = model(x)
            total_loss += float(self.loss_fn(out, y).item())
            total += int(y.numel())
        if was_training:
            model.train()
        return total_loss / max(total, 1)

    def score_round(
        self,
        global_model: torch.nn.Module,
        uploaded_models: Iterable[torch.nn.Module],
        uploaded_ids: Iterable[int],
    ) -> Dict[int, Dict[str, float | bool]]:
        base_loss = self._loss(global_model)
        rows: List[Tuple[int, float, float]] = []
        for cid, model in zip(uploaded_ids, uploaded_models):
            loss = self._loss(model)
            rows.append((int(cid), loss, loss - base_loss))

        scores = np.array([r[2] for r in rows], dtype=np.float64)
        if scores.size == 0:
            return {}
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        outlier_threshold = median + self.mad_k * mad

        out: Dict[int, Dict[str, float | bool]] = {}
        for cid, loss, score in rows:
            reject = bool(score > outlier_threshold and score > self.min_delta)
            out[cid] = {
                'reject': reject,
                'loss': float(loss),
                'score': float(score),
                'base_loss': float(base_loss),
                'median_score': median,
                'mad': mad,
                'outlier_threshold': outlier_threshold,
            }
        return out
