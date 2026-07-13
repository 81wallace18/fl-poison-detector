# fl-poison-detector

Detector de updates maliciosos em Federated Learning integrado ao PFLlibMonza. O fluxo ativo usa um MLP sobre features estatisticas, espectrais e contextuais dos pesos antes da agregacao FedAvg.

## Fluxo ativo

1. O MONZA cria um particionamento non-IID de MNIST ou CIFAR-10.
2. Uma execucao `cc=5` salva os updates dos clientes em `safetensors`.
3. `src/detector_mlp.py` treina e calibra o detector.
4. O MONZA compara o baseline `cc=3` com a defesa MLP `cc=7`.
5. Scripts e notebook geram FPR, FRR, recall por ataque e acuracia global.

O experimento DistilBERT foi encerrado porque apresentou pior custo e generalizacao. O codigo saiu do fluxo ativo; resultados e conclusoes permanecem em [RESULTS.md](docs/results/RESULTS.md), [EVOLUTION.md](docs/history/EVOLUTION.md) e [CC6_BERT_LIMITATION.md](docs/limitations/CC6_BERT_LIMITATION.md).

## Inicio rapido

O projeto usa `uv` e Python 3.12:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
bash scripts/check_project.sh
```

Confira a configuracao do CIFAR-10 sem alterar arquivos:

```bash
bash scripts/run_full.sh cifar10 --dry-run
```

Inicie o experimento completo:

```bash
GLOBAL_ROUNDS=300 TIMES=10 bash scripts/run_full.sh cifar10 --background
```

O passo a passo completo, inclusive os comandos manuais, esta em [HOWTO.md](docs/guides/HOWTO.md).

## Estrutura

```text
.
├── docs/                    # guia ativo, resultados, historico e limitacoes
├── notebooks/
│   └── notebook_monza_analysis.ipynb
├── src/                     # treino, inferencia e features do MLP
├── scripts/                 # workflows e auxiliares em estrutura plana
├── artifacts/               # resultados locais, ignorados pelo Git
└── PFLlibMonza/             # simulador FL integrado
```

Os quatro modulos usados pelo MONZA possuem copias em `src/` e em `PFLlibMonza/system/flcore/detector/`:

- `cc_mlp.py`
- `context_features.py`
- `features.py`
- `fl_save.py`

Depois de alterar um deles, execute:

```bash
python3 scripts/_check_runtime_sync.py
```

## Saidas

| Diretorio | Conteudo |
|---|---|
| `artifacts/runs/<dataset>/<run-id>/` | Log, CSVs, graficos e notebook executado. |
| `artifacts/models/<dataset>/mlp/` | Modelo, scaler, thresholds e diagnosticos. |
| `artifacts/dumps/<dataset>/` | Dumps temporarios usados no treinamento. |

`DetectionFPR/FRR` mede a decisao por round e e a metrica comparavel ao paper. `QuarantineFPR/FRR` mede a ocupacao acumulada da quarentena e deve ser tratada apenas como diagnostico.

## Documentacao

- [HOWTO.md](docs/guides/HOWTO.md): instalacao e execucao comando a comando.
- [MONZA_RESULTS.md](docs/results/MONZA_RESULTS.md): resultados experimentais em FL real.
- [EVOLUTION.md](docs/history/EVOLUTION.md): decisoes e evolucao metodologica.
- [scripts/README.md](scripts/README.md): referencia dos scripts ativos.
- [notebook_monza_analysis.ipynb](notebooks/notebook_monza_analysis.ipynb): analise reproduzivel sem outputs versionados.
