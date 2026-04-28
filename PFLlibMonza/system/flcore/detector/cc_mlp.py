"""ClientCheck variante MLP+features (cc=7 no MONZA).

Carrega o detector treinado em detector_mlp.py (artefatos em ARTIFACTS_DIR):
  model.pt        -- state_dict + input_dim + hidden + dropout + feature_names
  scaler.pkl      -- StandardScaler ajustado no treino

Reusa `features.extract_features` (60 features statisticas/espectrais/espaciais
sobre as 4 weight layers da FedAvgCNN). Funciona com qualquer state_dict que
contenha `conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`.

Roda em GPU por default.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn

# Garante import do features.py no mesmo diretorio
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from features import N_FEATURES, extract_features  # noqa: E402


class _MLPDetector(nn.Module):
    """Replica EXATA da arquitetura em detector_mlp.py:60. Mesma ordem de layers."""

    def __init__(self, input_dim: int = N_FEATURES, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden
        self.input_dim = input_dim
        self.hidden = list(hidden)
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClientCheckMLP:
    def __init__(
        self,
        artifacts_dir: str | os.PathLike,
        device: str | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        ckpt = torch.load(self.artifacts_dir / 'model.pt', map_location=self.device, weights_only=False)
        input_dim = ckpt.get('input_dim', N_FEATURES)
        hidden = tuple(ckpt.get('hidden', (128, 64)))
        dropout = float(ckpt.get('dropout', 0.3))
        self.model = _MLPDetector(input_dim=input_dim, hidden=hidden, dropout=dropout)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval().to(self.device)

        self.scaler = joblib.load(self.artifacts_dir / 'scaler.pkl')
        self.feature_names: List[str] = ckpt.get('feature_names', [])

    @torch.no_grad()
    def classify(self, state_dict: Mapping[str, torch.Tensor]) -> Dict:
        feats, _ = extract_features(state_dict, device=self.device)
        feats = feats.reshape(1, -1).astype(np.float32)
        feats_scaled = self.scaler.transform(feats).astype(np.float32)

        x = torch.from_numpy(feats_scaled).to(self.device)
        logits = self.model(x)[0]
        logit_ben = float(logits[0].item())
        logit_mal = float(logits[1].item())
        is_mal = logit_mal > logit_ben
        return {
            'label': int(is_mal),
            'is_malicious': bool(is_mal),
            'logit_ben': logit_ben,
            'logit_mal': logit_mal,
            'score': logit_mal - logit_ben,
        }

    def is_malicious(self, state_dict: Mapping[str, torch.Tensor]) -> bool:
        return self.classify(state_dict)['is_malicious']

    def filter_indices(self, state_dicts: Sequence[Mapping[str, torch.Tensor]]) -> List[int]:
        return [i for i, sd in enumerate(state_dicts) if not self.is_malicious(sd)]
