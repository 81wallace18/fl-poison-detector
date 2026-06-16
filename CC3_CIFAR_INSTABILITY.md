# cc3 (MONZA original) — instabilidade no CIFAR10 (pool starvation por quarentena 2ⁿ)

> **Status: DIAGNOSTICADO, sem fix aplicado** (decisão pendente). cc3 é o baseline MONZA do paper
> (cosine similarity + score + quarentena exponencial), **não** um detector deste trabalho.

## Resultado (paper-scenario: CNN, 100 clients, 30% mal, α=0.2, 300 rounds, 10 seeds)
| Dataset | cc3 (MONZA) acurácia | Paper Table 4 |
|---|---|---|
| MNIST | 97.6% ± 0.3 (estável) | 97.11 |
| **CIFAR10** | **33.4% ± 19.5 (instável)** | 47.58 |

**Bimodal no CIFAR** — os 10 seeds se dividem:
| seed | acc final | |
|---|---|---|
| 0,3,4,5,6,9 | 42–53% | OK (6 seeds) |
| **1,2,7,8** | **~9.8%** | **COLAPSARAM (4 seeds)** |

Todos os seeds ruins **chegam a 47–52% no meio do run e despencam num único round** (ex. seed 1:
0.500 no round 268 → 0.098 no 269). É **colapso catastrófico**, não degradação lenta.
Média dos 6 sobreviventes ≈ 48.6% ≈ paper; os 4 colapsos puxam a média pra 33.4%.

## Causa raiz: pool starvation pela quarentena exponencial
- `serveravg.py` (bloco cc==3, ~L341-407): remove cliente com `score < μ−σ` e chama `set_client_quarantine`.
- `set_client_quarantine` (~L211-213): `quarentena += 1; roundsQuarent = 2**quarentena` — o contador
  **nunca reseta** → bans de 2, 4, 8, 16, 32… rounds.
- `serverbase.select_clients` (L116-136): exclui quem tem `roundsQuarent > 0` do pool elegível.

Evidência (`analysis_outputs_cifar10/`):
- `QuarantineFPR` termina em **0.85–0.96** em quase todos os runs → 85–96% dos benignos presos.
- `cc_type_results_3.csv` (`Total` = modelos elegíveis/round): pool cai de 100 → single digits nos
  últimos ~40 rounds; nos seeds ruins chega a **1–4 clientes** por muitos rounds.
- Mecanismo: com α=0.2 non-IID, pool pequeno → FedAvg de 1–4 clientes vira modelo enviesado → no
  round seguinte quase todos parecem dissimilares → mais bans 2ⁿ → pool colapsa → modelo ~10%.
  Também: `μ−σ` remove ~16%/round **mesmo com todos benignos** (threshold estruturalmente agressivo).

## É fiel ao paper (não é bug nosso)
A implementação segue os Algoritmos 1 (quarentena 2ⁿ) e 2 (cosine + L2grad, threshold μ−σ). O paper
**admite a instabilidade** (Fig 5 "unstable peaks"; CIFAR-100 FPR 36.8%) mas **só reporta a média de
10 runs (sem desvio)** — o 47.58% dele provavelmente também mascara seeds colapsados, como nossos 6
sobreviventes (~48.6%) batem com o paper. **MNIST é estável** porque os gradientes convergem bem
agrupados, os scores cosine ficam separados e o pool nunca esvazia.

## Opções de correção (NÃO aplicadas)
1. **Cap + reset da quarentena** (`set_client_quarantine`): `roundsQuarent = min(2**quarentena, CAP)`
   (ex. 8–16) e zerar `quarentena` após um round limpo. Ataca o snowball diretamente.
2. **Piso no pool** (`select_clients`): nunca deixar o pool elegível abaixo de ~30–50% (liberar os
   menos-recentes quando faltar cliente). Evita a agregação com 1–4 clientes.
3. **Threshold mais suave** (`μ − k·σ`, k≈2–3) ou remover só uma fração limitada por round.
(1+2 são os de maior alavancagem; todos preservam a estrutura MONZA mas desviam do Algoritmo 1 exato.)

## Nota
Como o **cc7 (MLP+features) supera o cc3 com folga e é estável no CIFAR** (51% vs 33%, ±1.1 vs ±19.5),
a comparação já favorece o detector deste trabalho mesmo sem corrigir o cc3. Arquivos-chave:
`PFLlibMonza/system/flcore/servers/serveravg.py`, `serverbase.py`;
dados em `analysis_outputs_cifar10/{fpr_frr_results_3,cc_type_results_3}.csv` e h5 por seed.
