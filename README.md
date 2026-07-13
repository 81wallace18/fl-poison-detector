# fl-poison-detector

Detector binário de **updates maliciosos** em Federated Learning. Recebe um `state_dict` de cliente FL (uma `FedAvgCNN` com ~580k pesos) e classifica como **benigno** ou **malicioso** antes da agregação.

Duas abordagens implementadas e comparadas:
- **`detector.py`** — DistilBERT + LoRA sobre pesos discretizados + ramo tabular com features contextuais
- **`detector_mlp.py`** — MLP sobre features estatísticas dos pesos + delta local-global + validação pública limpa

## TL;DR

| Variante (`pretrained` × `hard`) | DistilBERT F1 | MLP F1 |
|---|---|---|
| 1. Leakage | 0.88 | **1.00** |
| 2. Hard | 0.89 | **0.96** |
| 3. **Pretrained + Hard** (mais realista) | 0.88 | **0.99** |
| 4. Pretrained + Easy | 0.86 | **1.00** |

MLP+features ganha por 0.10–0.15 F1 em todos os cenários.

Validação posterior em FL real (PFLlibMonza, 100 clientes Dirichlet non-IID, 30 maliciosos):

| cc | Defesa | QuarantineFPR | QuarantineFRR |
|---|---|---:|---:|
| 6 | NLP DistilBERT | 0.112 | 0.114 |
| **7** | **MLP+features** | **0.000** | **0.156** |

> ⚠️ **Estes são `QuarantineFPR/FRR`** (ocupação de quarentena, diagnóstico) — não a FPR/FRR de
> detecção do paper. A quarentena é exponencial (`2ⁿ`), então essa métrica faz *snowball* e satura;
> não é comparável à Table 4 do paper. A métrica headline (paper Eq 14/15) é a **detecção por-round
> `DetectionFPR/FRR`**, que só os runs novos logam (ex.: cc=3 MNIST 0.012/0.026, CIFAR10 0.066/0.060).
> DetectionFPR/FRR de cc2/6/7 está **pendente de re-run** com o logging atual.

MLP+features ficou como melhor detector final no fluxo normalizado. Detalhes em [`MONZA_RESULTS.md`](docs/results/MONZA_RESULTS.md).

Documentação:
- [`HOWTO.md`](docs/guides/HOWTO.md) — passo-a-passo do pipeline FL real (gera dataset com MONZA → treina detector → defesas cc=2/cc=3/cc=6/cc=7)
- [`MONZA_RESULTS.md`](docs/results/MONZA_RESULTS.md) — resultados experimentais em FL real
- [`RESULTS.md`](docs/results/RESULTS.md) — bench original 4×2 (dataset sintético)
- [`EVOLUTION.md`](docs/history/EVOLUTION.md) — como o projeto evoluiu
- [`notebook_monza_analysis.ipynb`](notebooks/notebook_monza_analysis.ipynb) — gráficos comparativos das defesas MONZA (`cc=2`/`cc=3`/`cc=6`/`cc=7`)

> ⚠️ **Limitações conhecidas (leia antes de comparar):**
> - [`CC3_CIFAR_INSTABILITY.md`](docs/limitations/CC3_CIFAR_INSTABILITY.md) — cc3 (MONZA original) colapsa em ~40% dos seeds no CIFAR (quarentena 2ⁿ → pool starvation); média 33±19.5 vs paper 47.6.
> - [`CC6_BERT_LIMITATION.md`](docs/limitations/CC6_BERT_LIMITATION.md) — cc6 (BERT) é cego a `shuffle` por design → colapsa; resultado negativo.

Valide a configuracao sem iniciar um experimento:

```bash
bash scripts/run_full_monza.sh --dry-run
bash scripts/run_full_cifar10.sh --dry-run
```

Run completo MONZA do zero:

```bash
SKIP_BERT=1 GLOBAL_ROUNDS=300 TIMES=10 \
bash scripts/run_full_monza.sh --background
```

Sweep curto de thresholds para melhorar `malicious_label` sem retreinar os detectores:

```bash
./scripts/sweep_monza_thresholds.sh --background
```

Veja [`scripts/README.md`](scripts/README.md) para separar fluxo principal, helpers e scripts legados.

O run completo usa por padrão `ROUND_INIT_ATK=5` e `DUMP_START_ROUND=6`: os rounds iniciais fazem warm-up limpo, depois o dump salva cada update junto com o modelo global anterior do round. `cc=6` e `cc=7` usam `combined_label_fpr05`, que combina score binario e head auxiliar de `malicious_label` com threshold calibrado na própria rodada, mantendo FPR benigno controlado. As features contextuais usam `PFLlibMonza/dataset/MNIST/public_val/`, separado do `test/` usado para avaliação.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python src/bench_grid.py
```

`bench_grid.py` faz tudo: treina baseline em MNIST, gera 4 variantes do dataset, roda os 2 detectores em cada uma, imprime tabela final + breakdown por ataque e salva o resultado em `artifacts/runs/synthetic/bench_grid_results.json`. ~30–40 min na RTX 5060 Ti.

Sempre executar a partir da **raiz do projeto**. Os paths configuráveis são
resolvidos a partir desse diretório.

O ambiente Python é único: use sempre `.venv/` na raiz. MONZA também deve ser executado com essa venv (`../../.venv/bin/python` quando o cwd for `PFLlibMonza/system`).

Nota sobre `malicious_label`: o runtime MONZA agora exige `PFLlibMonza/dataset/MNIST/train_mal/` para label flip real. O script `scripts/run_full_monza.sh` cria esse diretório automaticamente; para criar manualmente, use `scripts/create_label_flip_train_mal.py`.

## Estrutura

```text
.
├── README.md                              # entrada e visão geral
├── docs/                                 # guias, resultados, histórico e limitações
├── notebooks/                            # exploração e análise MONZA
├── src/
│   ├── detector.py                       # DistilBERT+LoRA sobre pesos→bins
│   ├── detector_mlp.py                   # MLP sobre features handcrafted
│   ├── features.py                       # extrator de 60 features de pesos
│   ├── context_features.py               # delta local-global + validação pública
│   ├── bench_grid.py                     # orquestrador 4×2
│   ├── cc.py                             # ClientCheck DistilBERT standalone
│   ├── cc_mlp.py                         # ClientCheckMLP — usado pelo MONZA
│   └── fl_save.py                        # helper de dump de state_dicts
├── scripts/
│   ├── workflows/                        # implementação dos experimentos
│   ├── tools/                            # preparação, análise e verificações
│   ├── legacy/                           # runners históricos
│   └── run_full_*.sh                     # wrappers públicos compatíveis
├── artifacts/                            # resultados locais, ignorados pelo Git
└── PFLlibMonza/                          # fork PFLlib (FL simulator) integrado
    └── system/flcore/detector/           # inferência MONZA: cc/cc_mlp/fl_save/features
```

Saídas geradas em runtime ficam sob `artifacts/` e não entram no Git:

| Diretório | Conteúdo |
|---|---|
| `artifacts/runs/<dataset>/<run-id>/` | Logs, CSVs, plots e análises de cada execução. |
| `artifacts/models/<dataset>/` | Detectores BERT e MLP treinados. |
| `artifacts/dumps/<dataset>/` | Dumps temporários de `state_dicts`. |
| `artifacts/cache/` | Datasets baixados pelo benchmark sintético. |
| `artifacts/archive/` | Resultados anteriores preservados e deduplicados. |

Os cinco módulos usados no runtime existem em `src/` e em
`PFLlibMonza/system/flcore/detector/`. Rode
`python3 scripts/check_runtime_sync.py` após qualquer alteração nesses arquivos.

## Documentação por arquivo

### `detector.py`

Pipeline DistilBERT+LoRA híbrido:

1. `preprocess_weights(state_dict)` — ordena camadas de forma canônica, pega tensores com `'weight'` no nome, normaliza cada um por quantis (q5/q95) com clamp em [0, 1], concatena, faz **pooling estratificado** via `torch.linspace(0, n-1, 512)` (em vez de truncamento), discretiza em 10000 bins (PAD_ID=0 reservado).
2. `extract_context_features(...)` monta sinais comportamentais para `malicious_label`: delta local-global, deltas da cabeça/classificador e métricas em validação pública MNIST limpa.
3. `tokenize_function` monta `input_ids` + `attention_mask` (1 para tokens não-PAD); as features contextuais entram em paralelo, normalizadas com `StandardScaler`.
4. `build_and_train(seed)` — DistilBERT base + LoRA `r=8` em `q_lin`/`v_lin`; o vetor `[CLS]` é concatenado com um ramo tabular (`LayerNorm -> MLP`) antes da classificação. O treino aceita `BERT_EPOCHS`, early stopping por `label_recall_fpr05`, oversampling de `malicious_label` e perda auxiliar ponderada.
5. `tune_threshold(...)` e `tune_combined_thresholds(...)` — calibram thresholds para FPR benigno controlado; `combined_label_fpr05` aplica regra OR entre score binario e head especializada em `malicious_label`.
6. `breakdown_by_type` — recall por tipo de ataque (`zeros`, `random`, `shuffle`, `malicious_label`).

`MODEL_SEED=15880` foi escolhido em experimento de ensemble como o que dá melhor F1 individual. Persistência em `FINAL_MODEL_DIR` + `metrics.json`.

### `detector_mlp.py`

Pipeline MLP:

1. `load_dataset()` — itera `state_dicts/*.safetensors` + `.json`, chama `extract_features`. Resultado: matriz X (N×60) + labels y + types.
2. `stratified_split(types, ...)` — `StratifiedShuffleSplit` por **tipo** de ataque (não só label) — garante que cada split tem amostras de cada categoria.
3. `StandardScaler` ajustado só no treino, persistido em `scaler.pkl`.
4. `MLPDetector` — `BatchNorm1d(60) → Linear(60→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→2)`. ~13k parâmetros.
5. Treino: AdamW lr=1e-3 wd=1e-4, scheduler `CosineAnnealingLR` por 60 epochs, batch=32, early stopping `patience=15` em F1 do eval, restaura best checkpoint.
6. Avaliação final + `breakdown_by_type` + `report.json` + `feature_names.json` em `ARTIFACTS_DIR`.

### `features.py`

Extrator puro (não tem treino). 4 camadas processadas: `conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`. Conv kernels viram matriz `(out, in·kH·kW)` para SVD/FFT 2D coerentes.

15 features por camada (× 4 camadas = 60):

| Categoria | Feature | O que mede |
|---|---|---|
| Magnitude | `l2`, `linf` | Norma Frobenius e máximo absoluto |
| Distribuição | `mean`, `std`, `kurt`, `zero_ratio`, `p5`, `p95` | Momentos e percentis |
| Entropia | `hist_entropy` | Entropia do histograma de 50 bins |
| Espectral | `sv1`, `sv2`, `sv3` | Top-3 singular values normalizados por Frobenius |
| Frequencial | `fft_hf_ratio` | Razão energia high-freq / low-freq via FFT-2D |
| Espacial | `tv` | Total variation média entre pesos vizinhos |
| Espacial | `autocorr1` | Autocorrelação Pearson lag-1 |

Roda na GPU se disponível (`torch.linalg.svdvals`, `torch.fft.fft2`). Custo ~ms por amostra.

### `bench_grid.py`

Orquestrador único que faz benchmark completo. Etapas:

1. Treina **`pretrained_base`** = `FedAvgCNN` em MNIST (10 epochs, ~1–2 min, cache em `mnist_data/`).
2. Gera 4 datasets em `state_dicts_grid/{1_leakage, 2_hard, 3_pretrained_hard, 4_pretrained_easy}/` (cada um N amostras de cada classe).
3. Roda `detector.py` e `detector_mlp.py` via subprocess para cada variante (8 treinos sequenciais), com env vars apontando pros dirs corretos. Streaming dos logs em tempo real, prefixados com `[variante DB|MLP]`.
4. Lê `metrics.json` / `report.json` de cada run, monta tabela final + breakdown por ataque e salva em `artifacts/runs/synthetic/bench_grid_results.json`.

### `notebooks/BertModelsclassify.ipynb`

Notebook de exploração + geração ad-hoc de dataset:
- **Cell 1**: definições (`FedAvgCNN`, ataques, helpers).
- **Cell 3**: **CONFIG + setup do `pretrained_base`** — flags `USE_PRETRAINED_BASE` (treina em MNIST se True), `HARDEN_ATTACKS` (ataques sutis se True), `N_SAMPLES_PER_CLASS`. Treina baseline conforme flag.
- **Cell 5**: gerador de `state_dicts/`, branching condicional pelo `HARDEN_ATTACKS`.

Útil pra gerar **um dataset específico** sem rodar o grid completo. Os flags do notebook reproduzem qualquer das 4 variantes do `bench_grid.py`.

## Configuração via env vars

| Variável | Usado por | Default | Descrição |
|---|---|---|---|
| `STATE_DICTS_DIR` | `detector.py`, `detector_mlp.py` | `state_dicts` | Pasta de leitura dos `.safetensors` |
| `FINAL_MODEL_DIR` | `detector.py` | `./detector_final` | Pasta de saída do modelo DistilBERT |
| `RUN_DIR` | `detector.py` | `./detector_runs/best` | `output_dir` do `Trainer` HF |
| `BERT_EPOCHS` | `detector.py` | `15` | Limite de epochs do DistilBERT |
| `BERT_EARLY_STOPPING_PATIENCE` | `detector.py` | `3` | Para quando `malicious_label` em dev nao melhora |
| `OVERSAMPLE_LABEL_FACTOR` | `detector.py`, `detector_mlp.py` | `1` | Replica amostras `malicious_label` no treino |
| `LABEL_LOSS_WEIGHT` | `detector.py`, `detector_mlp.py` | `1.0` | Peso da perda auxiliar de `malicious_label` |
| `BERT_MAX_BENIGN_FPR` | `detector.py` | `0.05` | Limite de FPR benigno para thresholds calibrados |
| `ARTIFACTS_DIR` | `detector_mlp.py` | `detector_mlp_artifacts` | Pasta de saída do MLP |
| `GRID_N_SAMPLES_PER_CLASS` | `bench_grid.py` | `1000` | Tamanho de cada variante do grid |
| `GRID_SKIP_DISTILBERT` | `bench_grid.py` | `0` | `1` pula DistilBERT (~3 min total só MLP) |

Exemplo — rodar grid rápido só com MLP:

```bash
GRID_N_SAMPLES_PER_CLASS=200 GRID_SKIP_DISTILBERT=1 .venv/bin/python src/bench_grid.py
```

Exemplo — rodar `detector.py` standalone num dataset custom:

```bash
STATE_DICTS_DIR=meus_dados FINAL_MODEL_DIR=./meu_modelo .venv/bin/python src/detector.py
```

## Reprodutibilidade

- Seeds fixas: `SEED=42` (data split) e `MODEL_SEED=15880` (treino do DistilBERT)
- `torch.backends.cudnn.deterministic=True` no MLP
- `bench_grid.py` é determinístico exceto pela parte de download do MNIST (a primeira vez)
- Resultado esperado bate com `artifacts/runs/synthetic/bench_grid_results.json` ±0.01 F1

## Limitações conhecidas

- **Apenas FedAvgCNN**: `features.py` e `detector.py` assumem 4 camadas com `weight` no nome (`conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`). Para outras arquiteturas, ajustar `LAYERS` em `features.py`.
- **Detecção isolada por update**: não usa comparação entre clientes (Krum/Multi-Krum/FoolsGold) nem trajetória multi-round (FLDetector). Defesa complementar, não substituta.
- **Threshold ainda e sensivel a split**: `detector.py` separa dev/calib/test por clientes; use `score_diagnostics.csv` para validar se `combined_label_fpr05` generalizou antes de comparar `cc=6`.
- **noise SNR alto é ceiling real**: contra benign treinado, ruído com SNR > 10 dB é ~indistinguível por construção.
