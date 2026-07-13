# Scripts

Scripts do pipeline MONZA/jpt. Rode todos a partir da raiz do repo.

## Comandos publicos

| Script | Uso |
|---|---|
| `run_full_monza.sh` | Regera MNIST, treina detectores e roda `cc=3`/`cc=7` (`cc=6` se `SKIP_BERT=0`). |
| `run_full_cifar10.sh` | Mesmo fluxo para CIFAR10. |
| `rerun_cc7.sh` | Reusa dataset existente, retreina MLP e roda apenas `cc=7`. |
| `sweep_monza_thresholds.sh` | Varre thresholds de detectores ja treinados. |

Os scripts `run_full_*` sao destrutivos para artefatos gerados: removem dumps, detectores, CSVs em `PFLlibMonza/system/` e resultados `.h5` do dataset alvo.

Os nomes acima sao wrappers estaveis. A implementacao fica em `workflows/`.
Use `--dry-run` para conferir o perfil e os caminhos sem modificar dados:

```bash
bash scripts/run_full_monza.sh --dry-run
bash scripts/run_full_cifar10.sh --dry-run
bash scripts/rerun_cc7.sh --dry-run
bash scripts/sweep_monza_thresholds.sh --dry-run
```

## Ferramentas

| Script | Uso |
|---|---|
| `create_label_flip_train_mal.py` | Cria `train_mal/` com label flip deterministico a partir de `train/`. |
| `plot_cc_attack_types.py` | Gera summaries/plots por tipo de ataque e FPR/FRR. |
| `summarize_threshold_sweep.py` | Consolida saídas de `sweep_monza_thresholds.sh`. |
| `_fpr_frr_io.py` | Normaliza CSVs antigos/novos de FPR/FRR. Importado por scripts de análise. |
| `check_runtime_sync.py` | Falha quando as copias de `src/` e do runtime MONZA divergem. |
| `check_project.sh` | Executa sintaxe, testes, links, notebooks e dry-runs. |

As implementacoes Python ficam em `tools/`; os nomes acima permanecem como
wrappers de compatibilidade.

## Legacy

| Script | Status |
|---|---|
| `legacy/create_train_mal.py` | Substituido por `create_label_flip_train_mal.py`. Mantido por historico. |
| `legacy/run_paper_cc3.sh` | Runner historico focado em `cc=3`. O fluxo atual usa `run_full_*`. |

## Verificacao

```bash
bash scripts/check_project.sh
```

Resultados, modelos, dumps e logs ficam em `artifacts/`, fora do Git.
