"""Feature extractor para detectar updates maliciosos em FL.

Extrai 15 features estatisticas/espectrais/espaciais por camada (4 camadas com
'weight' na FedAvgCNN) -> vetor de 60 features. Computa SVD e FFT na GPU.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

LAYERS: List[str] = [
    'conv1.0.weight',
    'conv2.0.weight',
    'fc1.0.weight',
    'fc.weight',
]
PREFIX: List[str] = ['conv1', 'conv2', 'fc1', 'fc']
FEATS: List[str] = [
    'l2', 'linf', 'mean', 'std', 'kurt', 'zero_ratio',
    'p5', 'p95', 'hist_entropy', 'sv1', 'sv2', 'sv3', 'fft_hf_ratio',
    'tv', 'autocorr1',
]
N_FEATURES_PER_LAYER = len(FEATS)
N_FEATURES = N_FEATURES_PER_LAYER * len(LAYERS)


def _kurtosis(x: torch.Tensor) -> torch.Tensor:
    # Fisher (excess) kurtosis, equivalente ao default do scipy.stats.kurtosis.
    mean = x.mean()
    diff = x - mean
    var = (diff ** 2).mean()
    if var.item() < 1e-20:
        return torch.tensor(0.0, device=x.device)
    m4 = (diff ** 4).mean()
    return m4 / (var ** 2) - 3.0


def _autocorr1(x: torch.Tensor) -> torch.Tensor:
    """Autocorrelacao Pearson lag-1 sobre o flatten.

    Pesos com estrutura espacial (treinados, ou com fan-in init estruturada)
    tendem a ter autocorr ~0.3-0.7 entre posicoes vizinhas. Shuffle quebra
    isso -> autocorr ~ 0.
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    m = x.mean()
    d = x - m
    num = (d[1:] * d[:-1]).sum()
    den = (d ** 2).sum().clamp_min(1e-12)
    return num / den


def _total_variation(M: torch.Tensor) -> torch.Tensor:
    """Total variation media: media de |M[i+1] - M[i]| sobre as duas dimensoes.

    Pesos suaves (treinados, ou com correlacao espacial em conv kernels)
    tem TV baixa; shuffle aumenta porque pesos vizinhos viram ruido.
    """
    if M.numel() < 2 or M.ndim < 2:
        return torch.tensor(0.0, device=M.device)
    tv_cols = (M[:, 1:] - M[:, :-1]).abs().mean()
    tv_rows = (M[1:, :] - M[:-1, :]).abs().mean()
    return (tv_cols + tv_rows) / 2.0


def _hist_entropy(x: torch.Tensor, bins: int = 50) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    if (hi - lo).item() < 1e-20:
        return torch.tensor(0.0, device=x.device)
    hist = torch.histc(x, bins=bins, min=float(lo), max=float(hi))
    p = hist / (hist.sum() + 1e-12)
    return -(p[p > 0] * torch.log(p[p > 0])).sum()


def _layer_feats(W: torch.Tensor) -> np.ndarray:
    if W.ndim == 4:
        # (out, in, kH, kW) -> (out, in*kH*kW) para SVD/FFT 2D coerentes.
        M = W.reshape(W.shape[0], -1)
    elif W.ndim == 2:
        M = W
    elif W.ndim > 1:
        M = W.reshape(W.shape[0], -1)
    else:
        M = W.unsqueeze(0)

    x = M.flatten()
    fro = torch.linalg.norm(x).clamp_min(1e-12)

    # Top-3 singular values (svdvals nao computa U, V -> mais rapido).
    sv_all = torch.linalg.svdvals(M)
    sv = torch.zeros(3, device=W.device)
    k = min(3, sv_all.numel())
    sv[:k] = sv_all[:k]

    ent = _hist_entropy(x, bins=50)

    # FFT-2D: razao de energia high-freq / low-freq.
    F = torch.fft.fft2(M)
    A = F.abs()
    h, w = A.shape
    hf = A[h // 2:, w // 2:].sum()
    lf = A[: h // 2, : w // 2].sum().clamp_min(1e-12)

    # Percentis: torch.quantile so aceita tamanhos < 16M; nossas camadas cabem.
    p5 = torch.quantile(x, 0.05)
    p95 = torch.quantile(x, 0.95)

    feats = torch.stack([
        fro,
        x.abs().max(),
        x.mean(),
        x.std(unbiased=False),
        _kurtosis(x),
        (x == 0).float().mean(),
        p5,
        p95,
        ent,
        sv[0] / fro,
        sv[1] / fro,
        sv[2] / fro,
        hf / lf,
        _total_variation(M),
        _autocorr1(x),
    ])
    return feats.detach().cpu().numpy().astype(np.float32)


def feature_names() -> List[str]:
    return [f'{pre}_{s}' for pre in PREFIX for s in FEATS]


def extract_features(state_dict, device: torch.device | None = None) -> Tuple[np.ndarray, List[str]]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats: List[np.ndarray] = []
    for key in LAYERS:
        W = state_dict[key].detach().to(device=device, dtype=torch.float32)
        feats.append(_layer_feats(W))
    return np.concatenate(feats), feature_names()
