# Evolução do projeto: lógica e metodologia

Como saímos de **F1=0.43 com bugs** para **F1=0.99 num benchmark realista**. Cada fase identificou um problema distinto — alguns eram bugs, outros decisões metodológicas, outros descobertas que reescreveram a abordagem.

## Ponto de partida

`detector.py` original usava **DistilBERT+LoRA** sobre os pesos discretizados em bins (cada peso vira um inteiro de 0 a 9999, vetor de 512 ints vira `input_ids`). Treino com 200 amostras (100 benignos + 100 maliciosos) por 3 epochs. Resultado: F1=0.43, confiança aleatória.

Diagnóstico inicial revelou múltiplos bugs e más decisões:

1. **`attention_mask` mascarava bin 0 válido** — `padding_id=0` colidia com o menor peso normalizado. Pesos no bin mínimo eram tratados como padding e ignorados.
2. **`BitsAndBytesConfig` com kwarg inválido** — `torch_dtype` não é argumento desse config; deveria ir no `from_pretrained`.
3. **Sem `if __name__ == '__main__'`** — importar `detector.py` num REPL disparava o treino.
4. **Truncamento brutal** dos pesos pra caber em 512 tokens — pegava os primeiros 512, descartando 99,9% (basicamente a `conv1` toda e nada de `conv2`/`fc1`/`fc`).
5. **Hiperparâmetros default do HuggingFace** — 3 epochs, lr=5e-5, sem weight decay nem scheduler, sem early stopping. Default é pra LLMs gigantes em datasets enormes; nosso caso é MLP-equivalente em 200 amostras.

## Fase 1 — fixes técnicos no DistilBERT

Corrigimos os bugs e ajustamos o pipeline. As mudanças com impacto:

- **Pooling estratificado** via `torch.linspace(0, n-1, 512)` em vez de truncamento — passa a representar todas as camadas (conv1+conv2+fc1+fc) de forma proporcional.
- **Normalização per-camada por quantis** (q5/q95) em vez de min/max global — cada camada mantém resolução em sua própria escala, e q5/q95 é robusto a outliers.
- **Hiperparâmetros realistas**: 15 epochs, lr=2e-4, weight_decay=0.01, scheduler cosine, warmup 6%, early stopping com `patience=7`, batch=16.
- **Threshold tuning pós-treino** (sweep de 200 thresholds em `logit_mal − logit_ben`).
- **Estrutura limpa** com `main()` + `if __name__`, seeds fixas (`SEED=42`, `MODEL_SEED=15880`).

Resultado: F1 **0.43 → 0.89** no dataset original (que ainda tinha leakage — voltaremos a isso).

Insight importante dessa fase: rodamos um **ensemble de 5 seeds** esperando ganho. Ensemble piorou o F1 (0.85 vs 0.89 do melhor individual). Conclusão: erros individuais são correlacionados, ensemble não ajudou. Ficamos com o melhor individual.

## Fase 2 — paradigma alternativo: MLP+features

Pesquisamos a literatura de FL Byzantine detection (FLDetector KDD'22, OptiGradTrust 2025, FedDMC, FLAIR, SVD+IsoForest 2024). Padrão claro: **features estatísticas/espectrais por camada → MLP** é o approach que funciona em FL.

Construímos `features.py` com 13 features por camada (52 totais, depois 60):

- Estatísticas: `l2`, `linf`, `mean`, `std`, `kurt`, `zero_ratio`, `p5`, `p95`, `hist_entropy`
- Espectrais: top-3 singular values normalizados por Frobenius (capturam shuffle), FFT high/low ratio (captura noise)

`detector_mlp.py` usa um MLP simples (60→128→64→2, ~13k parâmetros) sobre StandardScaler. Treino: AdamW, CosineAnnealingLR, early stopping em F1, em ~5s.

Resultado: **F1=1.00** com 0% FPR no dataset original. Aparentemente perfeito.

**Insight estrutural**: DistilBERT trata bin 100 e bin 101 como tokens não-relacionados — perde a ordinalidade dos valores. Com features explícitas, o MLP só precisa combinar — não precisa "redescobrir" relações entre pesos.

## Fase 3 — descoberta do leakage e endurecimento

F1=1.00 era **ilusão**. Análise cuidadosa do dataset revelou:

- **Os 200 maliciosos compartilhavam o mesmo `base_model`** (variável global na cell 3 do notebook). Cada um era apenas uma versão do `base_model` com um ataque aplicado.
- O `shuffle_model(base_model)` produzia 50 amostras com **stats globalmente idênticas** (mean, std, min, max — porque shuffle preserva tudo). Detectar isso era trivial: o modelo aprendia a "fingerprint" do `base_model` e flagava qualquer derivado.
- `random_param` substituía pesos por `U[0,1]` — distribuição com mean=0.5 e kurtose=1.8, drasticamente diferente da `kaiming_uniform_` de qualquer init. Trivial.

Endurecemos os ataques no notebook:

- **`fresh_base` por amostra**: cada malicioso parte de um `FedAvgCNN()` próprio (anti-leakage).
- **`random_smart`**: Gaussiano com sigma da própria camada. Preserva `mean/std/percentiles`; só quebra estrutura espacial.
- **`shuffle parcial`**: permuta uma fração `frac ∈ U[0.3, 1.0]` dos pesos por tensor.
- **`noise SNR variável`**: SNR uniforme `[3, 15] dB` (antes era fixo em 5dB; SNR=15 = ruído ~10% do sinal, muito sutil).

Resultado: ambos detectores **desabaram**.
- DistilBERT: F1 **0.89 → 0.43** (pior que aleatório!)
- MLP: F1 **1.00 → 0.86**, com `shuffle` recall = **0%**

`shuffle` em random init virou impossível com features estatísticas: TV, autocorrelação, momentos, percentis — todos invariantes (em expectativa) sob permutação de uma matriz random.

## Fase 4 — baseline realista

Pesquisa em mais profundidade (Yunis "Spectral Dynamics" 2024, FedLLM-Bench NeurIPS'24, "Permutation Invariant Functions" 2025) revelou:

- Em FL real, ataques **não acontecem no round 0** com modelo random. Ocorrem após 5–50 rondas, quando os pesos já têm estrutura espacial mensurável.
- Random init é artefato de benchmarks acadêmicos. Em deployment, baseline é sempre treinado.
- Yunis mostrou: após ~10 epochs em MNIST, singular values divergem do Marchenko-Pastur law, total variation cai >30%, autocorr lag-1 sobe pra >0.3.

Solução: **treinar `pretrained_base = FedAvgCNN` em MNIST por 10 epochs antes** de qualquer geração de amostras. Cada cliente FL (benigno e malicioso) parte desse modelo treinado. Variação entre amostras vem do ruído pequeno (proxy de 1 step local de SGD).

Adicionamos também 2 features espaciais que só fazem sentido com pesos estruturados:
- **`tv`** (total variation média) — pesos suaves (treinados) têm TV baixa; shuffle aumenta.
- **`autocorr1`** (Pearson lag-1) — treinado tem autocorr 0.3+; shuffle leva a 0.

Resultado: MLP **F1=0.97** no dataset realista (random init + hardened + pretrained). `shuffle` voltou pra **100%**, `noise` foi pra 80% (perde alguns SNR muito altos), os outros 100%.

## Fase 5 — bench grid 4×2

Pra mapear o espaço completamente, criamos `bench_grid.py` que cruza 2 flags ortogonais:

| | `HARDEN_ATTACKS=False` | `HARDEN_ATTACKS=True` |
|---|---|---|
| `USE_PRETRAINED_BASE=False` | **Leakage** (random init + ataques originais) | **Hard** (random init + ataques sutis) |
| `USE_PRETRAINED_BASE=True` | **Pretrained+Easy** (treinado + originais) | **Pretrained+Hard** (treinado + sutis) |

Subimos `N_SAMPLES_PER_CLASS` para 1000 (1600 train / 400 eval — intervalo de confiança ~3x mais apertado) e rodamos os 2 detectores em cada variante.

Resultados:

| Variante | DistilBERT F1 | MLP F1 |
|---|---|---|
| 1. Leakage | 0.88 | **1.00** |
| 2. Hard | 0.89 | **0.96** |
| 3. Pretrained+Hard | 0.88 | **0.99** |
| 4. Pretrained+Easy | 0.86 | **1.00** |

MLP venceu em todas as variantes por gap consistente de 0.10–0.15 F1. DistilBERT plateia perto de 0.88 independente do dataset — teto estrutural.

## Fase 6 — surpresa metodológica

Em fase 4 (200 amostras), MLP teve `shuffle=0%` no dataset hard sem pretreino. Concluímos com base em pesquisa que era "informacionalmente indistinguível".

Em fase 5 (1000 amostras), o mesmo cenário deu `shuffle=80%`.

**A claim teórica era direcionalmente correta mas absoluta demais.** Autocorr e TV de matrizes random shuffleadas têm flutuação estatística pequena (não exatamente zero) — abaixo do limiar de detecção com 200 amostras, vira sinal aproveitável com 1000. Lição: distinção informacional binária ("é detectável" vs "não é") esconde uma escala. Com dados suficientes, sinais sutis viram aproveitáveis.

## Decisões metodológicas relevantes

- **Split estratificado por tipo de ataque**, não só por label — garante que cada split tem amostras de cada categoria (benign + 4 ataques) na proporção correta. Sem isso, splits desbalanceados podem mascarar problemas de generalização.
- **Threshold tunado in-sample no `detector.py`** — sweep no eval set, otimista. Mantido por simplicidade; ganho marginal (~+0.01 F1).
- **MLP+features venceu DistilBERT por motivos estruturais**: tarefa numérica + dataset pequeno + features explícitas. DistilBERT é ferramenta errada; tokens-de-bins perdem ordinalidade.
- **Pretreino traz +0.03 F1 em regime amplo (1000 amostras)** mas é crítico em regime escasso (200). Em regime amplo, features estatísticas já discriminam bem mesmo sobre random init.
- **Limite informacional real**: noise com SNR alto contra benign treinado. Aproximadamente 80–92% recall máxima nessa categoria — não há feature pra capturar.

## Fora do escopo (futuro possível)

- **Krum / Multi-Krum / FoolsGold** — defesas que comparam updates entre clientes em vez de classificar isoladamente. Diferente paradigma; complementar.
- **Ataques mais sofisticados**: backdoor, label flipping, model-poisoning direcionado. O baseline atual cobre 4 ataques sintéticos; expansão é direta.
- **Arquiteturas maiores** (ResNet, transformer) — `features.py` precisaria adaptar `LAYERS`. Funções permanecem válidas.
- **Contrastive learning** entre rounds consecutivos — pega ataques on-off (cliente honesto por X rondas, ataca em Y).
- **Anchor-based defenses** com modelo pré-treinado público como referência.

## Lições

1. **F1=1.0 quase sempre indica leakage** ou benchmark inadequado. Desconfie sempre.
2. **Honestidade de threat model importa mais que sofisticação do modelo**. F1=0.99 no realista vale mais que F1=1.0 no leakage.
3. **Reaproveitar engenharia clássica de features** quando aplicável. SVD, FFT, autocorr resolveram problemas que DistilBERT não consegue por design.
4. **Distinções informacionais binárias mentem**. Tudo é questão de SNR vs número de amostras.
5. **Quando o paradigma errado, polir não resolve**. DistilBERT plateia em 0.88 independente do tuning. Feature engineering + MLP simples sobe pra 0.99 sem sweep.
