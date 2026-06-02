"""Targeted reverse-label-flip detector for MONZA client models.

This checker is intentionally separate from the previous behavior/LF checks.
It scores the specific attack used by this project, y -> num_classes - 1 - y,
with runtime-only signals available before aggregation: clean holdout behavior
and the client's classifier-head delta against the current global model.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class TargetedLabelFlipCheck:
    def __init__(
        self,
        val_loader,
        device: str | torch.device,
        num_classes: int = 10,
        min_score: float = 2.0,
        min_margin_delta: float = 0.05,
        min_loss_delta: float = -0.10,
        mad_k: float = 2.5,
        max_reject_fraction: float = 0.30,
        head_weight: float = 0.35,
        margin_weight: float = 1.0,
        loss_weight: float = 0.50,
        target_prob_weight: float = 0.50,
    ) -> None:
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.num_classes = int(num_classes)
        self.min_score = float(min_score)
        self.min_margin_delta = float(min_margin_delta)
        self.min_loss_delta = float(min_loss_delta)
        self.mad_k = float(mad_k)
        self.max_reject_fraction = float(max_reject_fraction)
        self.head_weight = float(head_weight)
        self.margin_weight = float(margin_weight)
        self.loss_weight = float(loss_weight)
        self.target_prob_weight = float(target_prob_weight)
        self.loss_none = nn.CrossEntropyLoss(reduction='none')

    def _to_device(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    @staticmethod
    def _final_weight_key(state_dict) -> str:
        for key in ('head.weight', 'fc.weight', 'base.fc.weight'):
            if key in state_dict:
                return key
        candidates = [k for k in state_dict if k.endswith('fc.weight') or k.endswith('head.weight')]
        if not candidates:
            raise KeyError('Nenhuma camada final encontrada no state_dict.')
        return candidates[-1]

    @staticmethod
    def _bias_key(weight_key: str, state_dict) -> str | None:
        bias_key = weight_key[:-len('weight')] + 'bias'
        return bias_key if bias_key in state_dict else None

    @torch.no_grad()
    def _class_behavior(self, model: torch.nn.Module) -> Dict[str, Dict[int, float]]:
        was_training = model.training
        model.eval()
        model.to(self.device)

        loss_totals: Dict[int, float] = {}
        margin_totals: Dict[int, float] = {}
        target_prob_totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}

        for x, y in self.val_loader:
            x, y = self._to_device(x, y)
            logits = model(x)
            losses = self.loss_none(logits, y)
            probs = torch.softmax(logits, dim=1)
            rows = torch.arange(y.numel(), device=y.device)
            targets = (self.num_classes - 1 - y).clamp(0, self.num_classes - 1)
            margins = logits[rows, targets] - logits[rows, y]
            target_probs = probs[rows, targets]

            for cls in torch.unique(y):
                mask = y == cls
                c = int(cls.item())
                loss_totals[c] = loss_totals.get(c, 0.0) + float(losses[mask].sum().item())
                margin_totals[c] = margin_totals.get(c, 0.0) + float(margins[mask].sum().item())
                target_prob_totals[c] = target_prob_totals.get(c, 0.0) + float(target_probs[mask].sum().item())
                counts[c] = counts.get(c, 0) + int(mask.sum().item())

        if was_training:
            model.train()

        return {
            'loss': {c: loss_totals[c] / max(counts[c], 1) for c in counts},
            'margin': {c: margin_totals[c] / max(counts[c], 1) for c in counts},
            'target_prob': {c: target_prob_totals[c] / max(counts[c], 1) for c in counts},
        }

    def _head_pair_scores(self, global_sd, client_sd, final_key: str) -> Dict[int, float]:
        weight_delta = (client_sd[final_key].detach() - global_sd[final_key].detach()).float()
        if weight_delta.ndim < 2 or weight_delta.shape[0] < self.num_classes:
            return {c: 0.0 for c in range(self.num_classes)}
        flat = weight_delta[: self.num_classes].reshape(self.num_classes, -1)
        row_norms = torch.linalg.norm(flat, dim=1).detach().cpu().numpy()

        bias_scores = np.zeros(self.num_classes, dtype=np.float64)
        bias_key = self._bias_key(final_key, global_sd)
        if bias_key is not None:
            bias_delta = (client_sd[bias_key].detach() - global_sd[bias_key].detach()).float()
            if bias_delta.numel() >= self.num_classes:
                bias_scores = bias_delta[: self.num_classes].detach().cpu().numpy().astype(np.float64)

        scores: Dict[int, float] = {}
        for c in range(self.num_classes):
            target = self.num_classes - 1 - c
            pair_norm = float(row_norms[target] + row_norms[c])
            target_minus_true_bias = float(bias_scores[target] - bias_scores[c])
            scores[c] = pair_norm + target_minus_true_bias
        return scores

    @staticmethod
    def _robust_z(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        arr = np.array(list(values.values()), dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        scale = max(1.4826 * mad, 1e-6)
        return {k: (float(v) - median) / scale for k, v in values.items()}

    def score_round(
        self,
        global_model: torch.nn.Module,
        uploaded_models: Iterable[torch.nn.Module],
        uploaded_ids: Iterable[int],
    ) -> Dict[int, Dict[str, float | bool | int]]:
        global_model = global_model.to(self.device)
        global_behavior = self._class_behavior(global_model)
        global_sd = global_model.state_dict()
        final_key = self._final_weight_key(global_sd)

        rows: List[Tuple[int, float, int, float, float, float, float]] = []
        for cid, model in zip(uploaded_ids, uploaded_models):
            model = model.to(self.device)
            behavior = self._class_behavior(model)
            client_sd = model.state_dict()
            head_scores = self._head_pair_scores(global_sd, client_sd, final_key)

            margin_deltas: Dict[int, float] = {}
            loss_deltas: Dict[int, float] = {}
            target_prob_deltas: Dict[int, float] = {}
            raw_scores: Dict[int, float] = {}
            for c, base_margin in global_behavior['margin'].items():
                margin_delta = float(behavior['margin'].get(c, base_margin) - base_margin)
                base_loss = global_behavior['loss'].get(c, 0.0)
                base_prob = global_behavior['target_prob'].get(c, 0.0)
                loss_delta = float(behavior['loss'].get(c, base_loss) - base_loss)
                target_prob_delta = float(behavior['target_prob'].get(c, base_prob) - base_prob)
                margin_deltas[c] = margin_delta
                loss_deltas[c] = loss_delta
                target_prob_deltas[c] = target_prob_delta

            head_z = self._robust_z(head_scores)
            margin_z = self._robust_z(margin_deltas)
            loss_z = self._robust_z(loss_deltas)
            target_prob_z = self._robust_z(target_prob_deltas)
            for c in margin_deltas:
                raw_scores[c] = (
                    self.margin_weight * margin_z.get(c, 0.0)
                    + self.loss_weight * loss_z.get(c, 0.0)
                    + self.target_prob_weight * target_prob_z.get(c, 0.0)
                    + self.head_weight * head_z.get(c, 0.0)
                )

            suspect_class, score = max(raw_scores.items(), key=lambda item: item[1])
            rows.append((
                int(cid),
                float(score),
                int(suspect_class),
                float(margin_deltas.get(suspect_class, 0.0)),
                float(loss_deltas.get(suspect_class, 0.0)),
                float(target_prob_deltas.get(suspect_class, 0.0)),
                float(head_scores.get(suspect_class, 0.0)),
            ))

        scores = np.array([r[1] for r in rows], dtype=np.float64)
        if scores.size == 0:
            return {}
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        outlier_threshold = median + self.mad_k * max(1.4826 * mad, 1e-6)

        out: Dict[int, Dict[str, float | bool | int]] = {}
        for cid, score, suspect_class, margin_delta, loss_delta, target_prob_delta, head_score in rows:
            target_class = self.num_classes - 1 - suspect_class
            score_outlier = bool(score > outlier_threshold and score > self.min_score)
            behavior_allowed = bool(
                margin_delta > self.min_margin_delta and loss_delta > self.min_loss_delta
            )
            reject = bool(score_outlier and behavior_allowed)
            out[cid] = {
                'reject': reject,
                'score_outlier': score_outlier,
                'behavior_allowed': behavior_allowed,
                'score': score,
                'suspect_class': suspect_class,
                'target_class': int(target_class),
                'margin_delta': margin_delta,
                'loss_delta': loss_delta,
                'target_prob_delta': target_prob_delta,
                'head_score': head_score,
                'median_score': median,
                'mad': mad,
                'outlier_threshold': outlier_threshold,
            }
        if self.max_reject_fraction > 0.0:
            max_rejects = max(1, int(np.ceil(len(rows) * self.max_reject_fraction)))
            rejected = [
                (cid, float(row['score']))
                for cid, row in out.items()
                if bool(row['reject'])
            ]
            if len(rejected) > max_rejects:
                keep = {
                    cid for cid, _ in sorted(rejected, key=lambda item: item[1], reverse=True)[:max_rejects]
                }
                for cid, row in out.items():
                    if bool(row['reject']) and cid not in keep:
                        row['reject'] = False
                        row['capped_by_max_reject_fraction'] = True
        else:
            for row in out.values():
                if bool(row['reject']):
                    row['reject'] = False
                    row['capped_by_max_reject_fraction'] = True
        return out
