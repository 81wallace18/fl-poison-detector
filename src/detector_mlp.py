"""Treino e avaliacao do detector MLP de updates maliciosos em FL.

Pipeline:
  state_dicts/*.safetensors  ->  features.extract_features (52 dims)
  -> StandardScaler          ->  MLPDetector (52->128->64->2)
  -> early stopping em F1    ->  artefatos em detector_mlp_artifacts/

Saida inclui breakdown de recall por tipo de ataque (benign + 4 maliciosos).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from features import N_FEATURES, extract_features, feature_names

SEED = 42
STATE_DICTS_DIR = os.environ.get('STATE_DICTS_DIR', 'state_dicts')
ARTIFACTS_DIR = Path(os.environ.get('ARTIFACTS_DIR', 'detector_mlp_artifacts'))
HIDDEN = (128, 64)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 15
TEST_SIZE = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPDetector(nn.Module):
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


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    files = sorted(glob.glob(os.path.join(STATE_DICTS_DIR, '*.safetensors')))
    assert files, f"Nenhum .safetensors em '{STATE_DICTS_DIR}/'."

    X_rows: List[np.ndarray] = []
    y_list: List[int] = []
    types: List[str] = []
    for f in tqdm(files, desc='extract features', unit='file'):
        sd = load_file(f)
        with open(f.replace('.safetensors', '.json')) as jf:
            meta = json.load(jf)
        feats, _ = extract_features(sd)
        X_rows.append(feats)
        y_list.append(int(meta['label']))
        types.append(meta['type'])

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, types


def stratified_split(types: List[str], test_size: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(types)), types))


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        'accuracy': accuracy_score(y_np, preds),
        'precision': precision_score(y_np, preds, zero_division=0),
        'recall': recall_score(y_np, preds, zero_division=0),
        'f1': f1_score(y_np, preds, zero_division=0),
        'preds': preds.tolist(),
    }


def breakdown_by_type(preds: np.ndarray, types_eval: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for t, p in zip(types_eval, preds):
        bucket = out.setdefault(t, {'total': 0, 'predicted_malicious': 0})
        bucket['total'] += 1
        bucket['predicted_malicious'] += int(p == 1)
    return out


def main() -> None:
    set_seed(SEED)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print('[1/4] Carregando state_dicts e extraindo features...')
    X, y, types = load_dataset()
    print(f'  Total: {len(y)} amostras | benignos={int((y == 0).sum())} | maliciosos={int((y == 1).sum())}')
    print(f'  Tipos: {sorted(set(types))}')
    print(f'  Feature dim: {X.shape[1]}')

    print('[2/4] Split estratificado por tipo de ataque (test_size=0.2)...')
    train_idx, eval_idx = stratified_split(types, TEST_SIZE, SEED)
    X_train, X_eval = X[train_idx], X[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]
    types_eval = [types[i] for i in eval_idx]
    print(f'  Train: {len(train_idx)} | Eval: {len(eval_idx)}')
    print(f'  Eval por tipo: {dict(zip(*np.unique(types_eval, return_counts=True)))}')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_eval_s = scaler.transform(X_eval).astype(np.float32)

    Xt_train = torch.from_numpy(X_train_s)
    yt_train = torch.from_numpy(y_train)
    Xt_eval = torch.from_numpy(X_eval_s)
    yt_eval = torch.from_numpy(y_eval)

    train_loader = DataLoader(
        TensorDataset(Xt_train, yt_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    print('[3/4] Treinando MLPDetector...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDetector(input_dim=N_FEATURES, hidden=HIDDEN, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    Xt_eval_dev = Xt_eval.to(device)
    yt_eval_dev = yt_eval.to(device)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        eval_metrics = evaluate(model, Xt_eval_dev, yt_eval_dev)
        f1 = eval_metrics['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f'  epoch {epoch:3d}/{EPOCHS} | train_loss={epoch_loss / max(n_batches, 1):.4f} '
                f"| eval f1={f1:.4f} acc={eval_metrics['accuracy']:.4f} "
                f"prec={eval_metrics['precision']:.4f} rec={eval_metrics['recall']:.4f} "
                f"| best_f1={best_f1:.4f}@{best_epoch}"
            )

        if epochs_without_improve >= PATIENCE:
            print(f'  early stopping na epoch {epoch} (sem melhora ha {PATIENCE} epochs)')
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    print('[4/4] Avaliacao final + breakdown por tipo de ataque')
    final = evaluate(model, Xt_eval_dev, yt_eval_dev)
    preds = np.array(final['preds'])
    print('\n--- Metricas binarias ---')
    print(f"  accuracy : {final['accuracy']:.4f}")
    print(f"  precision: {final['precision']:.4f}")
    print(f"  recall   : {final['recall']:.4f}")
    print(f"  f1       : {final['f1']:.4f}")
    print('\n--- classification_report ---')
    report_text = classification_report(
        yt_eval.numpy(), preds, target_names=['benign', 'malicious'], zero_division=0
    )
    print(report_text)

    print('--- Breakdown por tipo de ataque ---')
    by_type = breakdown_by_type(preds, types_eval)
    for t in sorted(by_type):
        b = by_type[t]
        ratio = b['predicted_malicious'] / b['total']
        kind = 'recall' if t != 'benign' else 'FPR'
        print(f"  {t:24s}: predicted_malicious={b['predicted_malicious']}/{b['total']} ({kind}={ratio:.2%})")

    print('\nSalvando artefatos em', ARTIFACTS_DIR)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'input_dim': N_FEATURES,
            'hidden': list(HIDDEN),
            'dropout': DROPOUT,
            'feature_names': feature_names(),
        },
        ARTIFACTS_DIR / 'model.pt',
    )
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.pkl')
    with open(ARTIFACTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names(), f, indent=2)

    report = {
        'best_epoch': best_epoch,
        'metrics': {k: final[k] for k in ('accuracy', 'precision', 'recall', 'f1')},
        'by_type': by_type,
        'config': {
            'seed': SEED,
            'hidden': list(HIDDEN),
            'dropout': DROPOUT,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'patience': PATIENCE,
            'test_size': TEST_SIZE,
        },
    }
    with open(ARTIFACTS_DIR / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nDONE. Best F1={best_f1:.4f} @ epoch {best_epoch}.')


if __name__ == '__main__':
    main()
