# HOWTO - pipeline MONZA -> dataset -> detector -> defesa

Este repo integra **PFLlibMonza** (simulador FL real em `PFLlibMonza/`) com os detectores em `src/`.

Fluxo principal:

1. MONZA gera o particionamento e dumpa updates de clientes como `.safetensors` + `.json`.
2. `src/detector_mlp.py` treina o detector MLP+features; DistilBERT e opcional.
3. MONZA roda as defesas e grava CSVs com `DetectionFPR/FRR`, `QuarantineFPR/FRR` e recall por tipo de ataque.

Resultados historicos ficam em [`MONZA_RESULTS.md`](../results/MONZA_RESULTS.md). Guia dos scripts em [`scripts/README.md`](../../scripts/README.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
```

O `--index-strategy unsafe-best-match` e necessario porque o arquivo de
requisitos combina o PyPI com o indice CUDA do PyTorch. Sem essa opcao, o
`uv` pode selecionar o indice CUDA para dependencias gerais e declarar a
resolucao impossivel.

Todos os comandos devem rodar da raiz do repo. Quando o cwd for `PFLlibMonza/system`, use `../../.venv/bin/python`.

Verificacao rapida:

```bash
cd PFLlibMonza/system
../../.venv/bin/python -c "
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

Esperado no host principal: `cuda: True`, GPU detectada e `imports OK`.

## Gerar Dataset Manualmente

Os scripts `run_full_*` ja fazem isto automaticamente. Use manualmente so para inspecao ou debug:

```bash
cd PFLlibMonza/dataset
../../.venv/bin/python generate_MNIST.py noniid - dir \
  --num-clients 100 \
  --dirichlet-alpha 0.2
cd ../..

.venv/bin/python scripts/create_label_flip_train_mal.py \
  --dataset-dir PFLlibMonza/dataset/MNIST \
  --num-classes 10

find PFLlibMonza/dataset/MNIST/train -name '*.npz' | wc -l       # esperado: 100
find PFLlibMonza/dataset/MNIST/train_mal -name '*.npz' | wc -l   # esperado: 100
```

Para CIFAR10, troque `generate_MNIST.py` por `generate_Cifar10.py` e `MNIST` por `Cifar10`.

`train_mal/` e obrigatorio para `malicious_label`: o runtime MONZA falha de proposito se ele nao existir.

## Fluxo Recomendado

Os comandos abaixo sao destrutivos para artefatos gerados: removem dumps, detectores, CSVs em `PFLlibMonza/system/` e resultados `.h5` do dataset alvo. Use quando quiser um experimento novo e limpo.

Antes de iniciar, valide o perfil e os caminhos sem alterar arquivos:

```bash
bash scripts/run_full_monza.sh --dry-run
bash scripts/run_full_cifar10.sh --dry-run
```

### MNIST Completo

```bash
SKIP_BERT=1 GLOBAL_ROUNDS=300 TIMES=10 \
bash scripts/run_full_monza.sh --background
```

### CIFAR10 Completo

```bash
SKIP_BERT=1 GLOBAL_ROUNDS=300 TIMES=10 \
bash scripts/run_full_cifar10.sh --background
```

Com `SKIP_BERT=1`, o fluxo roda: gerar dataset, criar `train_mal/`, dumpar updates com `cc=5`, treinar MLP, rodar baseline limpo/sem defesa, rodar `cc=3`, rodar `cc=7` e gerar summaries/plots. Ele nao roda `cc=2` nem `cc=6`.

### Reexecutar Apenas `cc=7`

Use depois de um full run, quando quiser reusar o dataset existente e retreinar apenas o MLP.

MNIST:

```bash
DATASET_NAME=MNIST GLOBAL_ROUNDS=300 TIMES=10 \
OVERSAMPLE_LABEL_FACTOR=4 LABEL_LOSS_WEIGHT=4 \
bash scripts/rerun_cc7.sh --background
```

CIFAR10:

```bash
DATASET_NAME=Cifar10 GLOBAL_ROUNDS=300 TIMES=10 \
OVERSAMPLE_LABEL_FACTOR=4 LABEL_LOSS_WEIGHT=4 \
bash scripts/rerun_cc7.sh --background
```

`rerun_cc7.sh` atualiza CSVs de `cc=7`, mas nao recompila todos os PNGs automaticamente. Para graficos atualizados, rode `plot_cc_attack_types.py` no dataset certo.

### Verificar Completude

```bash
python3 - <<'PY'
import csv, glob, collections

paths = glob.glob("artifacts/runs/*/*/analysis/fpr_frr_results_7.csv")
paths += glob.glob("PFLlibMonza/system/fpr_frr_results_7.csv")
for path in paths:
    rows = list(csv.DictReader(open(path)))
    by_run = collections.defaultdict(list)
    for row in rows:
        by_run[row["RunID"]].append(int(row["Round"]))
    print(path)
    for run_id, rounds in by_run.items():
        print(" ", run_id, min(rounds), max(rounds), len(rounds))
PY
```

Um run oficial deve chegar ao round `300` em cada seed. Se parar antes, trate como parcial.

## Variaveis Principais

| Variavel | Default | Efeito |
|---|---:|---|
| `GLOBAL_ROUNDS` | `50` nos scripts, `300` recomendado | Rounds do experimento final. |
| `TIMES` | `1` nos scripts, `10` recomendado | Numero de seeds/runs. |
| `SKIP_BERT` | `0` | `1` pula DistilBERT e `cc=6`. |
| `ROUND_INIT_ATK` | `5` | Rounds iniciais sem ataque. |
| `DUMP_START_ROUND` | `ROUND_INIT_ATK + 1` | Primeiro round salvo no dataset dos detectores. |
| `DUMP_GLOBAL_ROUNDS` | `60` | Rounds usados so para gerar dump de treino. |
| `KEEP_DUMP` | `0` | `1` preserva `state_dicts_monza_*` apos treinar. |
| `MLP_THRESHOLD_KEY` | `combined_label_fpr05` | Threshold do `report.json` usado pelo `cc=7`. |
| `MLP_THRESHOLD_VALUE` | vazio | Threshold binario manual para `cc=7`; sobrescreve `MLP_THRESHOLD_KEY`. |
| `PUBLIC_VAL_DIR` | dataset `public_val/` | Holdout limpo usado nas features contextuais. |
| `ARTIFACTS_ROOT` | `artifacts/` | Raiz local de runs, modelos e dumps. |
| `RUN_ID` | timestamp | Identificador usado no diretorio do run. |

## Pipeline Manual

Prefira `run_full_*`. O fluxo manual abaixo serve para debug.

### 1. Dump De Updates

```bash
cd PFLlibMonza/system
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
  -cc 5 -gr 60 -t 1 -ls 1 -did 0 -rfake 1 -ria 5 \
  --dump_state_dicts ../../artifacts/dumps/mnist/manual \
  --dump_start_round 6
cd ../..
```

Aceite:

```bash
find artifacts/dumps/mnist/manual -name '*.json' | wc -l
du -sh artifacts/dumps/mnist/manual
```

### 2. Treinar MLP

```bash
STATE_DICTS_DIR=./artifacts/dumps/mnist/manual \
PUBLIC_VAL_DIR=./PFLlibMonza/dataset/MNIST/public_val \
DATASET_NAME=MNIST \
ARTIFACTS_DIR=./artifacts/models/mnist/mlp \
.venv/bin/python -u src/detector_mlp.py
```

Artefatos esperados: `model.pt`, `scaler.pkl`, `feature_names.json`, `report.json`, `score_diagnostics.csv`.

### 3. Rodar Defesa `cc=7`

```bash
cd PFLlibMonza/system
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
  -cc 7 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 -ria 5 \
  --detector_dir ../../artifacts/models/mnist/mlp \
  --mlp_threshold_key combined_label_fpr05
cd ../..
```

Saidas principais:

- `PFLlibMonza/system/fpr_frr_results_7.csv`
- `PFLlibMonza/system/cc_detail_results_7.csv`
- `PFLlibMonza/system/cc_type_results_7.csv`

## Analise

### Summaries CLI

```bash
.venv/bin/python scripts/plot_cc_attack_types.py \
  --system-dir PFLlibMonza/system \
  --out-dir artifacts/runs/mnist/manual/analysis \
  --dataset MNIST \
  --tail-rounds 30
```

Para CIFAR10:

```bash
.venv/bin/python scripts/plot_cc_attack_types.py \
  --system-dir PFLlibMonza/system \
  --out-dir artifacts/runs/cifar10/manual/analysis \
  --dataset Cifar10 \
  --tail-rounds 30
```

### Media De FPR/FRR

```bash
python3 - <<'PY'
import glob
import sys
sys.path.insert(0, "scripts")
from _fpr_frr_io import load_fpr_frr, summarize_fpr_frr

for path in sorted(glob.glob("PFLlibMonza/system/fpr_frr_results_*.csv")):
    summary = summarize_fpr_frr(load_fpr_frr(path), min_round=5)
    print(path)
    print("  DetectionFPR={:.4f} DetectionFRR={:.4f}".format(
        summary["DetectionFPR_mean"], summary["DetectionFRR_mean"]))
    print("  QuarantineFPR={:.4f} QuarantineFRR={:.4f}".format(
        summary["QuarantineFPR_mean"], summary["QuarantineFRR_mean"]))
PY
```

`DetectionFPR/FRR` e a metrica comparavel ao paper. `QuarantineFPR/FRR` e diagnostico de ocupacao da quarentena.

### Sweep De Thresholds

Requer detectores treinados:

```bash
./scripts/sweep_monza_thresholds.sh --background
```

Saidas:

- `artifacts/runs/<dataset>/<run-id>_threshold_sweep/analysis/threshold_sweep_summary.csv`
- plots de recall, FPR, acuracia e trade-off no mesmo diretorio.

O sweep atual altera o threshold binario (`MLP_THRESHOLD_VALUE`/`BERT_THRESHOLD_VALUE`). Ele nao varre `label_threshold` separado da label head.

## Estrutura Relevante

```text
.
├── README.md
├── docs/
│   ├── guides/
│   ├── results/
│   ├── history/
│   └── limitations/
├── notebooks/
├── src/
│   ├── detector.py
│   ├── detector_mlp.py
│   ├── features.py
│   ├── context_features.py
│   ├── cc.py
│   ├── cc_mlp.py
│   └── fl_save.py
├── scripts/
│   ├── README.md
│   ├── run_full_monza.sh
│   ├── run_full_cifar10.sh
│   ├── rerun_cc7.sh
│   ├── sweep_monza_thresholds.sh
│   ├── create_label_flip_train_mal.py
│   ├── plot_cc_attack_types.py
│   ├── summarize_threshold_sweep.py
│   ├── _fpr_frr_io.py
│   ├── workflows/
│   ├── tools/
│   └── legacy/
├── artifacts/                 # local, ignorado pelo Git
└── PFLlibMonza/
    ├── dataset/
    └── system/
        ├── main.py
        └── flcore/detector/
```

`src/` e `PFLlibMonza/system/flcore/detector/` sao copias manuais. Se
mexer em um detector usado pelo MONZA, atualize os dois lados e valide:

```bash
python3 scripts/check_runtime_sync.py
```

## Troubleshooting

### `.venv/bin/python` nao existe

Crie a venv da raiz:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
```

### Usar um caminho de log personalizado

Os scripts criam o diretorio automaticamente. Para sobrescrever o caminho:

```bash
RUN_LOG=logs/manual.log bash scripts/run_full_monza.sh
```

### Dataset com numero errado de clientes

Regere com flags explicitas:

```bash
cd PFLlibMonza/dataset
../../.venv/bin/python generate_MNIST.py noniid - dir \
  --num-clients 100 \
  --dirichlet-alpha 0.2
cd ../..
```

Depois recrie `train_mal/`.

### Run parou no meio

```bash
ps -eo pid,etime,cmd | grep -E 'run_full|rerun_cc7|main.py|detector_mlp' | grep -v grep
find artifacts/runs -name run.log -print
df -h . PFLlibMonza/results /tmp
```

Se um `RunID` parar antes do round esperado, reexecute o comando inteiro. `-prev` em processo novo nao preserva estado de RNG entre seeds.

### Grafico nao mudou depois de `rerun_cc7.sh`

`rerun_cc7.sh` arquiva CSVs novos, mas nao recompila todos os PNGs. Reexecute `scripts/plot_cc_attack_types.py` para o dataset certo ou compare pelos CSVs.

### `cc7` bom no geral mas ruim em `malicious_label`

O recall geral pode ser dominado por `random`, `shuffle` e `zeros`. Confira `cc_detail_results_7.csv` filtrando `AttackType == malicious_label`.

Reexecute `cc=7` com a head auxiliar ativada:

```bash
DATASET_NAME=Cifar10 GLOBAL_ROUNDS=300 TIMES=10 \
OVERSAMPLE_LABEL_FACTOR=4 LABEL_LOSS_WEIGHT=4 \
bash scripts/rerun_cc7.sh --background
```

### Disco insuficiente

`-m VGG` em CIFAR10 pode gerar centenas de GB de dumps. Para debug, reduza `DUMP_GLOBAL_ROUNDS`, `GLOBAL_ROUNDS` ou `NUM_CLIENTS`.

## Referencias

- [`scripts/README.md`](../../scripts/README.md)
- [`README.md`](../../README.md)
- [`RESULTS.md`](../results/RESULTS.md)
- [`MONZA_RESULTS.md`](../results/MONZA_RESULTS.md)
- PFLlib upstream: https://github.com/TsingZ0/PFLlib
- MONZA fork: https://github.com/VeigarGit/PFLlibMonza
