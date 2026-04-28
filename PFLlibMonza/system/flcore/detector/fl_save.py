"""Salvamento de state_dicts FL no formato consumido por detector.py / detector_mlp.py.

Schema (espelha bench_grid.py:221-235):
  {out_dir}/{sample_id}.safetensors  -- pesos do cliente
  {out_dir}/{sample_id}.json         -- {"label": int, "type": str}

label: 0=benign, 1=malicious
type:  'benign' | 'malicious_zeros' | 'malicious_random' | 'malicious_shuffle' | 'malicious_label' | 'malicious_noise'

Tensores chegam em GPU (FL roda em CUDA); .detach().cpu() so na hora de serializar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from safetensors.torch import save_file


def save_client_update(
    state_dict: Mapping[str, torch.Tensor],
    label: int,
    type_: str,
    out_dir: str | os.PathLike,
    sample_id: str,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}

    safe_path = out / f'{sample_id}.safetensors'
    json_path = out / f'{sample_id}.json'

    save_file(cpu_sd, str(safe_path))
    with open(json_path, 'w') as f:
        json.dump({'label': int(label), 'type': str(type_)}, f)

    return safe_path


def save_round_dump(
    uploaded_models: Iterable,
    uploaded_ids: Iterable[int],
    clients_by_id: Dict[int, object],
    index_malicious,
    round_idx: int,
    out_dir: str | os.PathLike,
) -> int:
    """Dumpa todos os updates do round atual. Devolve quantos arquivos foram salvos.

    Para cada (model, client_id):
      - is_mal: prefere `client.is_malicious` (flag dinamica do round) ; fallback = client_id em index_malicious
      - type_:  prefere `client.last_attack_type`; fallback 'benign' / 'malicious_unknown'
    """
    index_malicious_set = set(int(x) for x in index_malicious) if index_malicious is not None else set()
    n_saved = 0
    for model, cid in zip(uploaded_models, uploaded_ids):
        client = clients_by_id.get(int(cid))
        is_mal_flag = getattr(client, 'is_malicious', None)
        if is_mal_flag is None:
            is_mal = int(cid) in index_malicious_set
        else:
            is_mal = bool(is_mal_flag)

        type_ = getattr(client, 'last_attack_type', None)
        if not type_:
            type_ = 'malicious_unknown' if is_mal else 'benign'

        sample_id = f'r{int(round_idx):03d}_c{int(cid):03d}_{type_}'
        save_client_update(model.state_dict(), int(is_mal), type_, out_dir, sample_id)
        n_saved += 1
    return n_saved
