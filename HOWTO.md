# HOWTO — pipeline completo (MONZA → dataset → detector → defesa)

Este repo integra **PFLlibMonza** (FL real, em `PFLlibMonza/`) com **jpt** (detectores NLP+MLP, em `src/`). O fluxo é:

1. **MONZA** roda FL com clientes maliciosos e dumpa state_dicts → dataset
2. **jpt** treina detectores (DistilBERT+LoRA e MLP+features) sobre o dataset
3. **MONZA** carrega o detector treinado num novo método de defesa (`cc=6` NLP ou `cc=7` MLP) e filtra clientes maliciosos antes da agregação

Resultado experimental fechado em [`MONZA_RESULTS.md`](MONZA_RESULTS.md). Análise visual em [`notebook_monza_analysis.ipynb`](notebook_monza_analysis.ipynb).

---

## Setup (1× só)

### 1. Clonar e instalar deps

```bash
git clone https://github.com/81wallace18/fl-poison-detector.git jpt
cd jpt

# venv pro jpt (treino dos detectores)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# venv pro MONZA (FL simulator)
cd PFLlibMonza
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
cd ..
```

> **GPU**: o pipeline assume CUDA (testado em RTX 5060 Ti, sm_120, torch 2.11.0+cu130). Se sua GPU é mais antiga, pode usar torch padrão (sem `--extra-index-url`), mas confira que `torch.cuda.is_available()` retorna True.

### 2. Gerar particionamento MNIST (100 clientes Dirichlet non-IID)

```bash
cd PFLlibMonza/dataset
python generate_MNIST.py noniid - dir
ls MNIST/train/ | wc -l   # esperado: 100
cd ../..
```

Scripts no `PFLlibMonza/dataset/` (`generate_MNIST.py`, `generate_Cifar10.py`, etc) hardcodam `num_clients` no topo do arquivo. Para outras configurações, edite a constante `num_clients`.

### 3. Verificar GPU e imports

```bash
cd PFLlibMonza/system
../.venv/bin/python -c "
import torch
from flcore.detector.cc import ClientCheck
from flcore.detector.cc_mlp import ClientCheckMLP
from flcore.detector import fl_save
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('imports OK')
"
cd ../..
```

Esperado: `cuda: True | GPU: NVIDIA GeForce RTX 5060 Ti` + `imports OK`.

---

## Pipeline (3 passos)

### Passo 1 — Gerar dataset rodando FL com dump

Em `PFLlibMonza/system/`, rodar simulação FL com `--dump_state_dicts <out_dir>`. Cada update de cliente em cada round vira um `.safetensors + .json` na pasta de dump.

```bash
cd PFLlibMonza/system
../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 5 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 \
    --dump_state_dicts ../../state_dicts_monza_cnn_mnist
cd ../..
```

| Flag | Significado |
|---|---|
| `-m CNN` | FedAvgCNN (ou `-m VGG` pra Cifar10) |
| `-data MNIST` | dataset PFLlib particionado |
| `-nc 100` | 100 clientes |
| `-nmc 30` | 30 são maliciosos (~30%) |
| `-atk all` | aleatoriamente sorteia entre {zero, random, shuffle, label} |
| `-cc 5` | sem defesa (modo "gerar dataset puro") |
| `-gr 50` | 50 rounds |
| `-rfake 1` | 100% chance do malicioso atacar a cada round |
| `--dump_state_dicts <dir>` | salva updates como `r{round:03d}_c{client:03d}_{type}.safetensors + .json` |

**Tempo**: ~15-25 min em RTX 5060 Ti. **Tamanho**: ~12 GB (FedAvgCNN ~580k params × 5100 amostras).

**Aceite**:
```bash
ls state_dicts_monza_cnn_mnist/*.json | wc -l                          # ~5000+
ls state_dicts_monza_cnn_mnist | grep -oP 'malicious_\w+|benign' | sort | uniq -c
du -sh state_dicts_monza_cnn_mnist
```

### Passo 2 — Treinar os detectores

Dois detectores em paralelo (paradigmas distintos pra comparação):

#### 2a — Detector NLP (DistilBERT+LoRA)

```bash
STATE_DICTS_DIR=./state_dicts_monza_cnn_mnist \
FINAL_MODEL_DIR=./detector_monza_cnn_mnist \
RUN_DIR=./detector_runs/monza_cnn_mnist \
.venv/bin/python -u src/detector.py
```

**Tempo**: ~15-30 min. **Saída**: `detector_monza_cnn_mnist/` (~3 MB com adapter LoRA + classifier head + `metrics.json`).

**Aceite**:
- `adapter_model.safetensors` deve ter ~3 MB (não 593 KB — se for menor, o head não foi salvo, ver bugfix abaixo).
- `metrics.json` com `default_argmax.f1` ≥ 0.80.

#### 2b — Detector MLP (60 features handcrafted)

```bash
STATE_DICTS_DIR=./state_dicts_monza_cnn_mnist \
ARTIFACTS_DIR=./detector_mlp_monza_cnn_mnist \
.venv/bin/python -u src/detector_mlp.py
```

**Tempo**: ~30 segundos (early stop em ~epoch 17). **Saída**: `detector_mlp_monza_cnn_mnist/` (~80 KB).

### Passo 3 — Defesa em produção (cc=6 NLP / cc=7 MLP)

Re-rodar FL com defesa ativada (`-cc 6` ou `-cc 7`) apontando pro detector treinado:

```bash
cd PFLlibMonza/system
# NLP
../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 6 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 \
    --detector_dir ../../detector_monza_cnn_mnist
# MLP
../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 7 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 \
    --detector_dir ../../detector_mlp_monza_cnn_mnist
cd ../..
```

**Saídas**: `PFLlibMonza/system/fpr_frr_results_6.csv` e `_7.csv`.

### Passo 3b (opcional) — Baselines do PFLlib

Pra comparar com defesas existentes (cosseno, cluster):

```bash
cd PFLlibMonza/system
# Cluster cosseno (cc=2)
../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 2 -gr 50 -t 1 -ls 1 -did 0 -rfake 1
# Cosseno + score (cc=3)
../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 3 -gr 50 -t 1 -ls 1 -did 0 -rfake 1
cd ../..
```

---

## Análise

### CLI rápida

```bash
cd PFLlibMonza/system
for csv in fpr_frr_results_{2,3,6,7}.csv; do
    echo "=== $csv ==="
    LC_NUMERIC=C tail -30 "$csv" | LC_NUMERIC=C awk -F, 'NF==3 && $1!~/Round/ {fpr+=$2; frr+=$3; n+=1} END {if(n>0) printf "  FPR_mean=%.4f  FRR_mean=%.4f  (n=%d)\n", fpr/n, frr/n, n}'
done
cd ../..
```

### Notebook visual

```bash
.venv/bin/jupyter notebook notebook_monza_analysis.ipynb
```

Gera 5 gráficos: FPR/FRR por round (4 defesas), trade-off scatter, métricas dos detectores, recall por tipo de ataque, sumário.

PNGs estáticos já estão no repo: `plot_*.png`.

---

## Estrutura do repo

```
jpt/
├── README.md                    # documentação original (DistilBERT vs MLP detectores)
├── EVOLUTION.md                 # narrativa do desenvolvimento (F1=0.43 → 0.99)
├── RESULTS.md                   # bench atual: 4×2 grid de variantes
├── MONZA_RESULTS.md             # 🆕 resultados experimentais MONZA
├── HOWTO.md                     # 🆕 este arquivo
├── notebook_monza_analysis.ipynb # 🆕 análise gráfica (4 defesas)
├── BertModelsclassify.ipynb     # gerador local de fallback (não-MONZA)
├── plot_*.png                   # 🆕 5 figuras geradas
├── src/
│   ├── detector.py              # treina DistilBERT+LoRA (modules_to_save=['pre_classifier','classifier'])
│   ├── detector_mlp.py          # treina MLP+features
│   ├── features.py              # 60 features estatisticas/espectrais (compativel com BaseHeadSplit do PFLlib)
│   ├── bench_grid.py            # bench 4×2 standalone (não usa MONZA)
│   ├── cc.py                    # 🆕 ClientCheck (DistilBERT) — usado pelo MONZA
│   ├── cc_mlp.py                # 🆕 ClientCheckMLP — usado pelo MONZA
│   └── fl_save.py               # 🆕 helper de dump de state_dicts
└── PFLlibMonza/                 # 🆕 fork do PFLlib (FL simulator)
    ├── system/
    │   ├── main.py              # +args --dump_state_dicts e --detector_dir
    │   ├── flcore/
    │   │   ├── attack/attack.py # ataques zeros/random/shuffle/label
    │   │   ├── clients/         # clientmaliciousavg.py expõe last_attack_type
    │   │   ├── servers/serveravg.py # +cases cc==6 (NLP) e cc==7 (MLP)
    │   │   ├── trainmodel/models.py # FedAvgCNN, VGG, etc
    │   │   └── detector/        # 🆕 cópia gêmea de cc.py, cc_mlp.py, fl_save.py, features.py
    │   ├── fpr_frr_results_*.csv # outputs por defesa (cc=2,3,6,7)
    │   └── run.sh
    ├── dataset/
    │   └── generate_*.py        # particionamento Dirichlet non-IID por dataset
    └── requirements.txt         # deps consolidadas (torch cu130 + transformers + peft + ...)
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'cvxpy'"

`PFLlibMonza/system/main.py` importa todos os servers no topo. `serverpac.py` requer cvxpy. Já está em `PFLlibMonza/requirements.txt`. Se faltar:

```bash
.venv/bin/pip install cvxpy
```

### "FileNotFoundError: '../dataset/MNIST/train/...'"

Dataset MNIST particionado pra `num_clients` errado. Edite `dataset/generate_MNIST.py:13` (`num_clients = 100`) e re-rode `python generate_MNIST.py noniid - dir`. O `check()` em `dataset/utils/dataset_utils.py` regera automaticamente se `num_clients` divergir do `config.json`.

### `cc=6` removendo todos os clientes (FPR=1.0)

Bug histórico: `LoraConfig` em `detector.py` precisava `modules_to_save=['pre_classifier', 'classifier']` pra persistir o head treinado. Já corrigido. Se reaparecer:

```bash
ls -la detector_monza_cnn_mnist/adapter_model.safetensors
# Esperado: ~3 MB. Se < 1 MB, o head não foi salvo — re-treinar.
```

### Disco insuficiente pra VGG/Cifar10

`-m VGG` gera state_dicts de ~56 MB cada → 100 clients × 50 rounds = 280 GB. Inviável sem amostragem. Opções:

- Reduzir rounds: `-gr 5` (~28 GB).
- Reduzir clientes: `-nc 30`.
- Adicionar amostragem no `fl_save.save_round_dump` (modificar pra dumpar 1 a cada N clientes/rounds).

### `ValueError: Input X contains infinity or a value too large for dtype('float32')`

Features explodem em ataques degenerados (`model_zeros` que vira `model_ones` no MONZA). `features.py:extract_features` já aplica `np.nan_to_num` no final pra sanitizar. Se reaparecer, conferir que o arquivo está atualizado.

---

## Referências

- Plano completo (sessão de desenvolvimento): `/home/wallace/.claude/plans/16-06-27-04-2026-rafael-veiga-lazy-dream.md`
- Bench atual standalone: ver `README.md` e `RESULTS.md` (rodam sem MONZA)
- PFLlib upstream: https://github.com/TsingZ0/PFLlib
- MONZA fork: https://github.com/VeigarGit/PFLlibMonza
