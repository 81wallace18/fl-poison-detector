# Scripts

Os scripts ativos ficam em uma unica pasta e devem ser executados a partir da raiz do repositorio.

## Comandos publicos

| Script | Uso |
|---|---|
| `run_full.sh` | Gera o dataset, treina o MLP, roda os baselines, `cc=3`, `cc=7` e as analises. |
| `rerun_cc7.sh` | Reusa o dataset existente, retreina o MLP e roda somente `cc=7`. |
| `create_label_flip_train_mal.py` | Cria `train_mal/` para o ataque de label flip. |
| `plot_cc_attack_types.py` | Gera CSVs e graficos de FPR, FRR, acuracia e recall por ataque. |
| `check_project.sh` | Valida scripts, Python, testes, links, runtime e notebook. |

Os workflows exigem um perfil explicito:

```bash
bash scripts/run_full.sh cifar10 --dry-run
bash scripts/run_full.sh mnist --dry-run
bash scripts/rerun_cc7.sh cifar10 --dry-run
```

Use `--background` para uma execucao longa. Sem uma opcao, o comando roda em primeiro plano.

`run_full.sh` remove artefatos gerados do dataset escolhido antes de iniciar. `rerun_cc7.sh` preserva o dataset e os resultados de `cc=3`/`cc=5`.

## Auxiliares internos

Arquivos com prefixo `_` nao sao comandos do fluxo normal:

| Arquivo | Responsabilidade |
|---|---|
| `_monza_common.sh` | Funcoes compartilhadas pelos dois workflows. |
| `_check_runtime_sync.py` | Confere as copias em `src/` e no runtime MONZA. |
| `_check_markdown_links.py` | Valida links locais da documentacao. |
| `_fpr_frr_io.py` | Normaliza layouts antigos e atuais dos CSVs de FPR/FRR. |

Resultados, modelos, dumps e logs sao gravados em `artifacts/`, que nao entra no Git.
