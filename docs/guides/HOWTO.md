# HOWTO - pipeline MONZA com MLP

Este guia parte de um clone limpo e usa somente o fluxo ativo: dataset real do MONZA, detector MLP e defesa `cc=7`. Execute todos os blocos a partir da raiz do repositorio, exceto quando o bloco contiver um `cd` explicito.

## 1. Clonar e instalar

```bash
git clone https://github.com/81wallace18/fl-poison-detector.git
cd fl-poison-detector
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
```

O indice adicional em `requirements.txt` fornece o PyTorch CUDA. `unsafe-best-match` permite que o `uv` resolva as demais dependencias pelo PyPI.

Valide o checkout:

```bash
bash scripts/check_project.sh
```

## 2. Conferir o workflow

Os perfis aceitos sao `cifar10` e `mnist`. O perfil e obrigatorio para evitar executar o dataset errado.

```bash
bash scripts/run_full.sh cifar10 --dry-run
bash scripts/rerun_cc7.sh cifar10 --dry-run
```

O dry-run confere a sincronizacao do runtime e imprime todos os caminhos sem gerar dataset ou modelo.

## 3. Executar CIFAR-10 completo

```bash
GLOBAL_ROUNDS=300 TIMES=10 \
bash scripts/run_full.sh cifar10 --background
```

O comando imprime o PID e o caminho do log. Acompanhe com:

```bash
tail -f artifacts/runs/cifar10/*/run.log
```

O workflow executa, nesta ordem:

1. gera o particionamento CIFAR-10 non-IID;
2. cria `train_mal/` para label flip;
3. roda `cc=5` para criar os dumps de treinamento;
4. treina e calibra o MLP;
5. roda os baselines limpo e sem defesa;
6. roda `cc=3` e `cc=7`;
7. executa o notebook e os summaries CLI.

Para MNIST, use o mesmo comando com o perfil `mnist`.

## 4. Validacao rapida

Este comando percorre o pipeline completo com escala reduzida. Ele valida integracao e formatos, mas nao produz metricas cientificas comparaveis ao experimento oficial.

```bash
GLOBAL_ROUNDS=2 TIMES=1 \
DUMP_GLOBAL_ROUNDS=8 DUMP_TIMES=1 \
NUM_CLIENTS=10 NUM_MALICIOUS=4 \
ROUND_INIT_ATK=0 DUMP_START_ROUND=1 \
MONZA_EXPECTED_ROWS=2 MONZA_TAIL_ROUNDS=2 \
bash scripts/run_full.sh cifar10
```

## 5. Reexecutar somente cc=7

Use depois de um full run. O comando reutiliza o dataset existente, recria o dump, retreina o MLP e substitui somente as saidas de `cc=7`.

```bash
GLOBAL_ROUNDS=300 TIMES=10 \
bash scripts/rerun_cc7.sh cifar10 --background
```

## 6. Pipeline manual CIFAR-10

Use estes comandos para depurar uma etapa especifica. O workflow da secao anterior ja executa todos eles.

### Gerar o dataset

```bash
cd PFLlibMonza/dataset
../../.venv/bin/python generate_Cifar10.py noniid - dir \
  --num-clients 100 \
  --dirichlet-alpha 0.2
cd ../..

.venv/bin/python scripts/create_label_flip_train_mal.py \
  --dataset-dir PFLlibMonza/dataset/Cifar10 \
  --num-classes 10

find PFLlibMonza/dataset/Cifar10/train -name '*.npz' | wc -l
find PFLlibMonza/dataset/Cifar10/train_mal -name '*.npz' | wc -l
```

Os dois contadores devem imprimir `100`.

### Criar dumps para treinamento

```bash
cd PFLlibMonza/system
../../.venv/bin/python main.py \
  -m CNN -data Cifar10 -nmc 30 -nc 100 -jr 1 \
  -atk all -ria 5 -cc 5 -gr 60 -t 1 -ls 1 -did 0 -rfake 1 \
  --dump_state_dicts ../../artifacts/dumps/cifar10/manual \
  --dump_start_round 6
cd ../..

find artifacts/dumps/cifar10/manual -name '*.json' | wc -l
du -sh artifacts/dumps/cifar10/manual
```

### Treinar o MLP

```bash
STATE_DICTS_DIR=./artifacts/dumps/cifar10/manual \
PUBLIC_VAL_DIR=./PFLlibMonza/dataset/Cifar10/public_val \
DATASET_NAME=Cifar10 \
ARTIFACTS_DIR=./artifacts/models/cifar10/mlp \
.venv/bin/python -u src/detector_mlp.py
```

Artefatos esperados:

```bash
ls artifacts/models/cifar10/mlp/{model.pt,scaler.pkl,feature_names.json,report.json,score_diagnostics.csv}
```

### Rodar a defesa cc=7

```bash
cd PFLlibMonza/system
PUBLIC_VAL_DIR=../dataset/Cifar10/public_val DATASET_NAME=Cifar10 \
../../.venv/bin/python main.py \
  -m CNN -data Cifar10 -nmc 30 -nc 100 -jr 1 \
  -atk all -ria 5 -cc 7 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 \
  --detector_dir ../../artifacts/models/cifar10/mlp \
  --mlp_threshold_key combined_label_fpr05
cd ../..
```

Saidas esperadas:

```bash
ls PFLlibMonza/system/{fpr_frr_results_7.csv,cc_detail_results_7.csv,cc_type_results_7.csv}
```

### Gerar analises

```bash
.venv/bin/python scripts/plot_cc_attack_types.py \
  --system-dir PFLlibMonza/system \
  --out-dir artifacts/runs/cifar10/manual/analysis \
  --dataset Cifar10 \
  --tail-rounds 30

REPO_ROOT="$PWD" \
DATASET_NAME=Cifar10 \
ANALYSIS_OUT="$PWD/artifacts/runs/cifar10/manual/analysis" \
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/notebook_monza_analysis.ipynb \
  --output notebook-monza-analysis.executed.ipynb \
  --output-dir artifacts/runs/cifar10/manual
```

## 7. Variaveis principais

| Variavel | Default | Efeito |
|---|---:|---|
| `GLOBAL_ROUNDS` | `50` | Rounds de cada experimento de avaliacao. |
| `TIMES` | `1` | Quantidade de execucoes/seeds. |
| `NUM_CLIENTS` | `100` | Clientes no particionamento e na simulacao. |
| `NUM_MALICIOUS` | `30` | Clientes pertencentes ao grupo malicioso. |
| `ROUND_INIT_ATK` | `5` | Ultimo round de warm-up sem ataques. |
| `DUMP_GLOBAL_ROUNDS` | `60` | Rounds usados para criar o dataset do detector. |
| `DUMP_START_ROUND` | `ROUND_INIT_ATK + 1` | Primeiro round salvo no dump. |
| `KEEP_DUMP` | `0` | Use `1` para preservar os dumps apos o treino. |
| `MLP_THRESHOLD_KEY` | `combined_label_fpr05` | Regra calibrada carregada do `report.json`. |
| `MLP_THRESHOLD_VALUE` | vazio | Sobrescreve manualmente o threshold binario. |
| `ARTIFACTS_ROOT` | `artifacts/` | Raiz de modelos, dumps, logs e analises. |

## 8. Verificar resultados

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path('PFLlibMonza/system/fpr_frr_results_7.csv')
rows = list(csv.DictReader(path.open()))
run_id = rows[-1]['RunID']
rounds = [int(row['Round']) for row in rows if row['RunID'] == run_id]
print('run:', run_id, 'rounds:', len(set(rounds)), 'ultimo:', max(rounds))
PY
```

No experimento oficial, o ultimo round deve ser `300`. `DetectionFPR/FRR` e a metrica por decisao; `QuarantineFPR/FRR` e apenas diagnostico acumulado.

## 9. Organizacao dos scripts

```text
scripts/
├── README.md
├── check_project.sh
├── run_full.sh
├── rerun_cc7.sh
├── create_label_flip_train_mal.py
├── plot_cc_attack_types.py
├── _check_markdown_links.py
├── _check_runtime_sync.py
├── _fpr_frr_io.py
└── _monza_common.sh
```

Consulte [scripts/README.md](../../scripts/README.md) para a responsabilidade de cada arquivo.
