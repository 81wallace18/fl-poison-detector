# cc6 (DistilBERT) — limitação confirmada: cego a `shuffle` (e label)

> Registro historico do motivo pelo qual o BERT e o `cc=6` foram retirados do fluxo executavel.

## Resultado
No cenário do paper (CNN, 100 clients, 30% mal, α=0.2, 300 rounds, 10 seeds), a defesa **cc6
(detector NLP DistilBERT+LoRA) colapsa** o modelo global: acurácia final ~10% (random) tanto em
**MNIST (10.65% ± 0.71)** quanto em **CIFAR10 (9.87% ± 0.19)** — igual a "sem defesa". As outras
defesas recuperam (MNIST cc3=97.6%, cc7=97.8%).

## Causa raiz (confirmada empiricamente)
Run de confirmação (cc6 MNIST, 1 seed, 300 rounds, reusando `detector_monza_cnn_mnist`):

| round | acc global | shuffle removidos | zeros | random |
|---|---|---|---|---|
| 6 (1º round de ataque, ria=5) | **70.1%** | **0/10 (0%)** | 9/9 | 3/7 |
| 7 | **9.2%** ⬇️ | **0/5 (0%)** | 4/4 | 4/5 |
| 8–300 | **10.3% (congelado)** | — | — | — |

**Mecanismo:** o BERT pega `zeros` (100%) e parte de `random`, mas **NÃO pega `shuffle` (0%)** nem
`label`. No round 6 passam ~10 shuffle + 4 random → a agregação FedAvg corrompe o modelo num único
round (70%→9%) → o modelo vira degenerado → nesse modelo morto **todos os updates ficam parecidos**,
o detector não separa mais nada (DetFPR=0, QuarFPR=0) → permanece em ~10% para sempre.
**Não é cascata de quarentena** (QuarFPR=0 após o colapso).

## Por que o BERT não vê `shuffle`
1. **Representation ~invariante a permutação:** `src/detector.py:preprocess_weights` normaliza por
   camada, concatena e faz *pooling estratificado* (linspace sobre pesos ordenados) → bins/tokens.
   O ataque `shuffle` é uma **permutação de pesos por camada** (`attack.py:shuffle_model`,
   `Pin/Pout`), que preserva o histograma/estatística por camada → some no pooling ordenado.
2. **Score não separável (evidência):** nos rounds 6–7, `BERTScore` benign ∈ [−1.43, −0.14]
   (média −0.66) e shuffle ∈ [−0.57, **+0.01**] (média −0.28) — **sobrepostos**. Threshold binário
   = +0.072. Baixar o threshold pra pegar shuffle marcaria muito benigno junto (FPR alto → recolapso).
   Logo **não é ajuste de threshold; é o representation que não carrega o sinal de shuffle.**
3. Os `context_features` (delta vs global, queda de acurácia no public_val) *deveriam* flagar shuffle,
   mas o head híbrido treinado não os converte em score separável — então mesmo com eles o BERTScore
   de shuffle continua na faixa benigna.

Isto confirma a conclusão anterior do projeto ("DistilBERT é desperdício pra essa tarefa"): o cc7
(MLP+features: SVD/FFT/total-variation/autocorr) **pega shuffle** porque suas features capturam
estrutura espacial/posicional; o cc6 não.

## Decisão
- **cc6 fica como resultado negativo** no comparativo (não entra como defesa funcional).
- **Não** aplicar "cap de remoção" — o cc6 remove de MENOS, não de mais; o cap pioraria.

## Nota de pesquisa (TODO — detector que se encaixe)
Procurar/desenhar um detector que detecte `shuffle` (permutação) e `label`:
- Feature **posição-aware** (ex.: correlação posicional pré/pós, assinatura por-índice da camada),
  já presentes no `cc7` (`features.py`: total-variation, autocorr lag-1, top-k SVD) — daí o cc7 pegar.
- Ou fundir um **check estrutural pré-agregação** (delta-vs-global enorme + norma por-camada
  preservada mas reordenada) como sinal explícito anti-shuffle.
- Ou um detector que use os `context_features` (queda de acurácia no holdout limpo) com peso forte —
  shuffle/label derrubam a acurácia no public_val, sinal robusto e independente do representation.
- Referência de evidência: `PFLlibMonza/system/cc_detail_results_6.csv` (BERTScore por tipo),
  trajetória de colapso no log do run de confirmação.
