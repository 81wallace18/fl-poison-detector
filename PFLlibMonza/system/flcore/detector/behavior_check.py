"""Behavior-based label-flip detector for MONZA client models.

The MLP and BERT detectors inspect weight fingerprints. Label-flip updates can
keep those fingerprints close to benign updates, so this checker evaluates the
uploaded model on a clean public holdout and searches for flip-target margins.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class BehaviorLabelFlipCheck:
    def __init__(
        self,
        val_loader,
        device: str | torch.device,
        num_classes: int = 10,
        min_margin_delta: float = 0.20,
        min_loss_delta: float = -0.05,
        mad_k: float = 3.0,
        max_reject_fraction: float = 0.05,
        flip_mode: str = 'reverse',
    ) -> None:
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.num_classes = int(num_classes)
        self.min_margin_delta = float(min_margin_delta)
        self.min_loss_delta = float(min_loss_delta)
        self.mad_k = float(mad_k)
        self.max_reject_fraction = float(max_reject_fraction)
        self.flip_mode = str(flip_mode)
        self.loss_none = nn.CrossEntropyLoss(reduction='none')
        if self.flip_mode not in ('reverse', 'max_non_true'):
            raise ValueError("flip_mode deve ser 'reverse' ou 'max_non_true'.")

    def _to_device(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    @torch.no_grad()
    def _metrics(self, model: torch.nn.Module) -> Dict[str, object]:
        was_training = model.training
        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_correct = 0
        total = 0
        margin_totals: Dict[int, float] = {}
        margin_counts: Dict[int, int] = {}
        pred_counts = np.zeros(self.num_classes, dtype=np.int64)

        for x, y in self.val_loader:
            x, y = self._to_device(x, y)
            logits = model(x)
            losses = self.loss_none(logits, y)
            preds = logits.argmax(dim=1)
            rows = torch.arange(y.numel(), device=y.device)
            if self.flip_mode == 'reverse':
                flipped = (self.num_classes - 1 - y).clamp(0, self.num_classes - 1)
                margins = logits[rows, flipped] - logits[rows, y]
            else:
                masked = logits.clone()
                masked[rows, y] = float('-inf')
                margins = masked.max(dim=1).values - logits[rows, y]

            total_loss += float(losses.sum().item())
            total_correct += int((preds == y).sum().item())
            total += int(y.numel())

            bincount = torch.bincount(preds.detach().cpu(), minlength=self.num_classes)
            pred_counts += bincount.numpy()[: self.num_classes]

            for cls in torch.unique(y):
                mask = y == cls
                c = int(cls.item())
                margin_totals[c] = margin_totals.get(c, 0.0) + float(margins[mask].sum().item())
                margin_counts[c] = margin_counts.get(c, 0) + int(mask.sum().item())

        if was_training:
            model.train()

        class_margins = {
            c: margin_totals[c] / max(margin_counts[c], 1)
            for c in margin_totals
        }
        return {
            'loss': total_loss / max(total, 1),
            'accuracy': total_correct / max(total, 1),
            'class_margins': class_margins,
            'pred_counts': pred_counts,
        }

    def score_round(
        self,
        global_model: torch.nn.Module,
        uploaded_models: Iterable[torch.nn.Module],
        uploaded_ids: Iterable[int],
    ) -> Dict[int, Dict[str, float | bool | int]]:
        base = self._metrics(global_model)
        base_margins = base['class_margins']
        rows: List[Tuple[int, float, float, int, float, float]] = []

        for cid, model in zip(uploaded_ids, uploaded_models):
            metrics = self._metrics(model)
            class_margins = metrics['class_margins']
            deltas = {
                c: float(class_margins.get(c, base_margin) - base_margin)
                for c, base_margin in base_margins.items()
            }
            worst_class, worst_margin_delta = max(deltas.items(), key=lambda item: item[1])
            loss_delta = float(metrics['loss'] - base['loss'])
            accuracy_delta = float(metrics['accuracy'] - base['accuracy'])
            rows.append((
                int(cid),
                float(worst_margin_delta),
                loss_delta,
                int(worst_class),
                float(metrics['accuracy']),
                accuracy_delta,
            ))

        margin_scores = np.array([r[1] for r in rows], dtype=np.float64)
        if margin_scores.size == 0:
            return {}
        median = float(np.median(margin_scores))
        mad = float(np.median(np.abs(margin_scores - median)))
        outlier_threshold = median + self.mad_k * max(mad, 1e-6)
        if self.max_reject_fraction > 0.0:
            q = min(max(1.0 - self.max_reject_fraction, 0.0), 1.0)
            outlier_threshold = max(outlier_threshold, float(np.quantile(margin_scores, q)))

        out: Dict[int, Dict[str, float | bool | int]] = {}
        for cid, margin_delta, loss_delta, worst_class, accuracy, accuracy_delta in rows:
            margin_outlier = bool(
                margin_delta > outlier_threshold and margin_delta > self.min_margin_delta
            )
            loss_allowed = bool(loss_delta > self.min_loss_delta)
            reject = bool(margin_outlier and loss_allowed)
            out[cid] = {
                'reject': reject,
                'margin_outlier': margin_outlier,
                'loss_allowed': loss_allowed,
                'worst_class': worst_class,
                'margin_delta': margin_delta,
                'loss_delta': loss_delta,
                'accuracy': accuracy,
                'accuracy_delta': accuracy_delta,
                'median_margin_delta': median,
                'mad': mad,
                'outlier_threshold': outlier_threshold,
            }
        return out
