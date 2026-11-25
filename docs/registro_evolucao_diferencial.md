# Registro Técnico e Experimental — Evolução Diferencial, DETO, DASE e Estudos de Ablação

**Branch:** `evolucao-diferencial`  
**Data:** 23–24 de novembro de 2026  
**Objetivo:** Documentar todas as implementações, decisões metodológicas, experimentos, análises estatísticas e descobertas realizadas para subsidiar o relatório final da disciplina de mestrado em Computação Evolucionária.

---

## 1. Motivação e Visão Geral

Nos experimentos iniciais do projeto, o Algoritmo Genético (AG) apresentou desempenho abaixo do esperado frente a baselines online simples (como `B4` — Seleção Online) e métodos da literatura (`DASE`). Para investigar se o gargalo estava no algoritmo evolucionário utilizado ou na formulação do processo de adaptação pós-drift, executamos uma investigação sistemática em 5 fases:

1. **Implementação da Evolução Diferencial (DE)** como otimizador de primeira classe no simplex $\Delta^M$;
2. **Integração de `DETO` e `DASE`** ao mesmo motor de replay prequencial sobre o cache de matrizes de probabilidade `P`, garantindo comparação justa na mesma entrada;
3. **Diagnóstico experimental (`n_min = 100` vs `n_min = 500`)** com análise de detectores (`ADWIN`, `DDM`, `Oráculo`) e sensibilidade;
4. **Estudos de ablação completos** para **DE** (21.960 execuções) e **AG** (27.360 execuções);
5. **Validação final otimizada** em 30 sementes (6 bases + DETO oficial nas 5 sintéticas).

---

## 2. Alterações de Código e Arquitetura

### 2.1. Núcleo Evolucionário (`src/evolution/`)
- **`src/evolution/base.py`** *(novo)*: extraiu os tipos e utilitários compartilhados (`FitnessFn`, `OptResult`, `random_simplex`, `mean_pairwise_distance`) para eliminar acoplamento entre AG e DE.
- **`src/evolution/de.py`** *(novo)*: implementou `DEConfig` e `DifferentialEvolution` com reparo no simplex $\Delta^M$ e mesma interface `run(pop0, fitness, rng, hypermutation=False) -> OptResult` do AG:
  - **8 variantes clássicas**: mutação `{rand/1, best/1, current-to-best/1, rand/2}` × cruzamento `{bin, exp}`;
  - **Dither opcional em $F$**: amostragem $F \sim U(0.5, 1.0)$ por geração;
  - **Hipermutação pós-drift (`S4`)**: além de elevar o fator de escala para `hyper_F = 0.9`, adiciona uma perturbação gaussiana suave (`hyper_sigma`) durante `hyper_gens = 10` gerações para escapar do colapso populacional ($x_{r2} - x_{r3} \approx 0$) em populações *warm-start* já convergidas.
- **`src/evolution/ga.py`** e **`src/evolution/__init__.py`**: refatorados para usar `base.py` e exportar `DEConfig`, `DE_STRATEGIES` e `DifferentialEvolution` mantendo compatibilidade total com imports anteriores.

### 2.2. Motor de Replay e Baselines (`src/evaluation/`, `src/baselines/`)
- **`src/evaluation/replay.py`**:
  - Adicionado `de: DEConfig` em `ReplayConfig` e suporte nativo a `optimizer="de"` com `cfg.de.pop_size`;
  - Adicionadas `run_deto_replay` (DETO com transferência entre drifts acionado pelo detector + buffer de warning) e `run_dase_replay` (DASE original com detector HCDD + RBFN + DE em chunks periódicos) sobre as matrizes `P` em cache.
- **`src/baselines/deto_adapter.py`**:
  - Mantida integralmente a **implementação oficial do DETO** (sem alteração no algoritmo ou na acumulação de tarefas-fonte);
  - Silenciados `print`s internos (`why are your arrays not F order?`) e `RuntimeWarning`s de L-BFGS do `GPy` (sem efeito numérico);
  - Adicionada proteção contra `numpy.linalg.LinAlgError` (*"not positive definite, even with jitter"*) dentro do laço de BO (`model.fit` / `optimizer.optimize`), encerrando graciosamente a iteração de BO quando a matriz de covariância do GP fica numericamente singular e preservando o melhor vetor de pesos já encontrado, evitando perda da semente inteira.

### 2.3. Scripts Experimentais e Notebooks
- **`scripts/run_experiments.py`**: atualizado com `--optimizer {ga,de}`, `--de-strategy`, `--de-F`, `--de-CR` e `--de-dither`.
- **`scripts/run_de_deto_dase.py`** *(novo)*: script completo para rodar `de`, `ga`, `deto`, `dase`, `baselines`, `benchmarks`, `detector` e `sensitivity` com:
  - Barra de progresso `tqdm` separada em construção de cache e experimentos;
  - Controle automático de threads BLAS (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) para evitar *oversubscription* em execuções paralelas (`--workers 12`);
  - Salvamento incremental por `(dataset, seed)` em `checkpoints/`, retomada automática e isolamento de exceções por job (`errors.log`);
  - Parâmetros configuráveis de ablação/otimização (`--n-min`, `--de-hyper-sigma`, `--ga-blx-alpha`, etc.).
- **`scripts/run_de_ablation.py`** *(novo)* e **`notebooks/ablacao_de.ipynb`** *(novo)*: estudo de ablação completo do DE (8 estratégias × `S1..S6` + hiperparâmetros `F`/dither, `CR`, `pop_size`, `hyper_sigma`).
- **`scripts/run_ga_ablation.py`** *(novo)* e **`notebooks/ablacao_ga.ipynb`** *(novo)*: estudo de ablação completo do AG (6 variantes estruturais × `S1..S6` + hiperparâmetros `tournament_k`, `blx_alpha`, `pc`, `sigma`, `pop_size`, `hyper_sigma`).
- **`notebooks/analise_de_deto_dase.ipynb`** *(novo)*: análise completa com tabelas descritivas, boxplots, Friedman + Nemenyi (diagramas CD incluindo todos os baselines), Wilcoxon pareado + Holm + $\hat{A}_{12}$, curvas temporais, convergência pós-drift, bloco `detector` (H5) e `sensitivity`.
- **`notebooks/analise_bases_reais.ipynb`** *(novo)*: análise dedicada exclusivamente às bases reais (`Elec2`), incluindo matriz heatmap de tamanho de efeito $\hat{A}_{12}$.
- **`tests/test_core.py`**: expandido para **27 testes unitários** (todos passando), cobrindo todas as 8 estratégias de DE, escape de população colapsada com hipermutação, replay com DE e testes smoke de `run_deto_replay` e `run_dase_replay`.

---

## 3. Protocolo Experimental e Estatístico

- **Bases avaliadas (6 streams, 30 sementes = 180 blocos `base × semente`)**:
  - *Recorrentes*: `SEA-Rec` (50k inst., 7 drifts), `STAGGER-Rec` (30k inst., 5 drifts)
  - *Novos conceitos*: `AGR-Abr` (50k inst., 4 drifts abruptos), `AGR-Grad` (50k inst., 4 drifts graduais de 2.000 inst.)
  - *Incremental*: `RBF-Inc` (50k inst.)
  - *Real*: `Elec2` (45.312 inst.)
- **Métrica principal**: Macro-F1 prequencial (`macro_f1`), complementada por acurácia, kappa, macro-F1 em janela deslizante (`wf1_mean`), queda máxima (`max_drop`), tempo de recuperação (`recovery_time`), avaliações (`n_evals`), `gens_to_95` e tempo de replay.
- **Testes estatísticos** (`src/evaluation/stats.py`):
  - **Friedman + Nemenyi (Demšar, 2006)** a $\alpha = 0{,}05$ com Diagramas de Diferença Crítica ($CD = q_{0{,}05}(k)\sqrt{k(k+1)/(6N)}$);
  - **Wilcoxon pareado** por semente dentro de cada base com **correção de Holm**;
  - **Tamanho de efeito $\hat{A}_{12}$ de Vargha-Delaney** (`<0,06` desprezível, `<0,14` pequeno, `<0,21` médio, `≥0,21` grande).
- **Hardware**: Intel Core i5-14600K (14 núcleos / 20 threads), executando com `--workers 12` e threads BLAS fixadas em 1.

---

## 4. Resultados e Descobertas por Fase

### Fase 1 — Comparação Inicial (`n_min = 100`, 30 sementes, `results/de_dase_completo`)
- **Total**: 10.080 execuções (3.420 principais + blocos `detector` e `sensitivity`).
- **Desempenho global (`macro_f1`)**: `B4` (`0.8764`, rank `2.828`) > `DASE` (`0.8659`, rank `3.889`) > `B7` (`0.8584`, rank `4.611`) ≈ `GA-S1` (`0.8576`, rank `4.728`) ≈ `DE-S4` (`0.8563`, rank `4.700`).
- **Diagnóstico científico do gargalo**:
  1. Trocar AG por DE, isoladamente, manteve empate técnico — o gargalo **não** era o otimizador;
  2. Na análise de sensibilidade, **`n_min = 200`** foi o fator de maior ganho (`+0.0026`) e **`n_min = 50`** o pior (`-0.0029`);
  3. No bloco `detector` (H5), o **Oráculo foi pior** que `ADWIN` e `DDM` (`STAGGER-Rec`: `0.7690` vs `0.8013` do DDM; `AGR-Abr`: `0.8363` vs `0.8470`), pois disparava no instante exato do drift e otimizava pesos sobre apenas 100 instâncias transitórias enquanto o pool base ainda não havia se estabilizado.

---

### Fase 2 — Nivelamento do Conjunto de Avaliação (`n_min = 500`, 30 sementes, `results/de_dase_nmin500`)
Igualando o tamanho mínimo do conjunto de avaliação $E$ de DE e AG à janela de 500 instâncias utilizada por `B4` e `DASE`:
- **Salto expressivo em todos os evolucionários (+1,3 a +1,8 p.p.)**:
  - `GA-S6`: `0.8572` → **`0.8737`** (`+0.0165`)
  - `GA-S4`: `0.8567` → **`0.8732`** (`+0.0165`)
  - `DE-S4`: `0.8563` → **`0.8713`** (`+0.0150`)
  - `DE-S1`: `0.8560` → **`0.8694`** (`+0.0134`)
- **Friedman ($N = 180$, $CD = 0{,}895$, $p = 6{,}5 \times 10^{-140}$)**: `B4` (`3.111`), `GA-S6` (`3.156`), `B7` (`3.539`) e `DE-S4` (`3.606`) formaram o bloco vencedor empatado, **superando `DASE` (`4.950`), `B6` (`5.578`) e `B3` (`5.617`) com significância estatística**.
- **Confirmação em H5**: com `n_min = 500`, o Oráculo saltou de `0.7690` → `0.8253` em `STAGGER-Rec` e passou a liderar em `AGR-Abr` (`0.8626`), `AGR-Grad` (`0.8516`) e `SEA-Rec` (`0.8530`).

---

### Fase 3 — Estudo de Ablação da Evolução Diferencial (21.960 execuções, `results/de_ablation`)
- **8 estratégias DE × `S1..S6` (Friedman $N = 1080$, $p = 3{,}2 \times 10^{-69}$, $CD = 0{,}319$)**:
  - **Mais robustas globalmente** (não colapsam em `S2`/`S5`): `rand/2/exp` (rank `3.997`), `rand/1/exp` (`4.016`), `rand/2/bin` (`4.080`), `rand/1/bin` (`4.156`);
  - **Melhor combinação absoluta**: **`current-to-best/1/exp + S4`** (`0.8735`, rank `3.956` em S4, $p = 0{,}011$);
  - **Melhor custo-benefício**: `best/1/exp + S1` (`0.8729` com apenas `7.7` gerações até 95% e `3377` avaliações);
  - O cruzamento **exponencial (`/exp`)** superou consistentemente o binomial (`/bin`).
- **Ablação de hiperparâmetros**:
  - **`hyper_sigma` em `S4`** *(achado principal)*: sem perturbação extra (`0.0`), `current-to-best/1/bin` caiu para `0.8585` (`div_end = 0.030`); com **`hyper_sigma = 0.1`**, subiu para **`0.8734`** (**`+0.0149`**), provando que aumentar só `F` não escapa de populações convergidas ($x_{r2}-x_{r3}\approx 0$);
  - **`CR`**: `0.3` (`0.8729`) e `0.5` (`0.8726`) superaram `0.9` (`0.8699`) em `current-to-best/1/bin` por preservarem diversidade;
  - **`F` / `dither`**: `dither` (`0.8715`) e `F = 0.8` (`0.8714`) superaram `F = 0.3` (`0.8690`) em `current-to-best/1/bin`.

---

### Fase 4 — Estudo de Ablação do Algoritmo Genético (27.360 execuções, `results/ga_ablation`)
- **Ablação estrutural (Friedman $N = 1080$, $p = 7{,}3 \times 10^{-80}$, $CD = 0{,}229$)**:
  - **Mutação gaussiana é indispensável**: `no_mutation` (`pm=0`) derrubou a média de `0.8724` → **`0.8556`** (rank `4.496`), colapsando em `S2` (`0.8403`), `S4` (`0.8403`) e `S5` (`0.8405`);
  - **Crossover BLX-$\alpha$** contribuiu de forma consistente (`no_crossover`: `0.8709` vs `gen_e2`: `0.8724`);
  - `gen_e2` (`0.8724`, melhor combinação `gen_e2 + S6 = 0.8737`), `gen_e0` (`0.8723`), `gen_e5` (`0.8720`) e `mu+lambda` (`0.8715`) empataram estatisticamente; `mu+lambda` reduziu fortemente a diversidade final (`div_end = 0.028` vs `0.104`).
- **Ablação de hiperparâmetros**:
  - **`blx_alpha = 0.25`** no `generational` alcançou o maior valor do estudo (**`0.8742`** vs `0.5 = 0.8734`);
  - `tournament_k = 3`, `pc = 0.9`, `sigma = 0.1`, `pop_size = 30` e `hyper_sigma = 0.5` confirmaram-se como pontos ótimos.

---

### Fase 5 — Validação Final Otimizada + DETO Oficial (30 sementes)

**Configuração final otimizada (`results/otimizado_nmin500`, 10.440 execuções):**
- **DE**: `current-to-best/1/exp`, `dither=True`, `CR=0.5`, `hyper_sigma=0.1`, `pop_size=30`, `n_min=500`;
- **AG**: `generational (e=2)`, `tournament_k=3`, `blx_alpha=0.25`, `pc=0.9`, `sigma=0.1`, `pop_size=30`, `hyper_sigma=0.5`, `n_min=500`;
- **DETO oficial (`results/deto_nmin500`, 150 execuções)**: avaliado nas **5 bases sintéticas × 30 sementes** com `n_min=500`, `n_init=12`, `n_iter=15` (omitido em `Elec2` devido ao custo $O(S^4)$ do GP exato com dezenas de drifts do ADWIN).

#### Tabela Comparativa por Base (Macro-F1 médio, 30 sementes)

| Base | `GA-S6` | `DE-S3` | `DE-S4` | `DETO` | `DASE` | `B4` | `B6` | `B3` | `B7` | `B1` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AGR-Abr` | `0.8887` | `0.8799` | `0.8820` | `0.8798` | `0.8780` | **`0.8890`** | `0.8737` | `0.8736` | `0.8844` | `0.8488` |
| `AGR-Grad` | `0.8789` | `0.8745` | `0.8714` | `0.8582` | `0.8695` | **`0.8854`** | `0.8661` | `0.8660` | `0.8762` | `0.8408` |
| `RBF-Inc` | `0.9088` | `0.9094` | `0.9094` | `0.9086` | `0.8969` | `0.9098` | `0.9099` | **`0.9101`** | `0.9091` | `0.8243` |
| `SEA-Rec` | `0.8550` | `0.8554` | `0.8565` | `0.8531` | **`0.8566`** | `0.8528` | `0.8413` | `0.8406` | `0.8561` | `0.8448` |
| `STAGGER-Rec` | `0.8512` | `0.8496` | `0.8487` | `0.8423` | `0.8275` | **`0.8652`** | `0.8511` | `0.8512` | `0.8464` | `0.8122` |
| **Média 5 Sintéticas** | **`0.8765`** | `0.8738` | `0.8736` | `0.8684` | `0.8657` | `0.8804` | `0.8684` | `0.8683` | `0.8744` | `0.8342` |
| **`Elec2` (Real)** | `0.8700` | **`0.8704`** | `0.8703` | — | `0.8669` | `0.8565` | `0.8470` | `0.8453` | `0.8664` | `0.8408` |
| **Média Geral (6 bases)** | `0.8754` | `0.8732` | `0.8731` | — | `0.8659` | **`0.8764`** | `0.8648` | `0.8645` | `0.8731` | `0.8353` |
| **Tempo replay / stream** | `0.36 s` | `0.46 s` | `0.44 s` | `538.6 s` | `4.69 s` | `0.07 s` | `0.02 s` | `0.01 s` | `0.59 s` | `0.002 s` |

#### Principais Conclusões Estatísticas (6 bases, $N = 180$, $CD = 0{,}895$, $p = 8{,}6 \times 10^{-151}$)
1. **Ranking de Friedman**: **`GA-S6` ficou em 1º lugar absoluto (`rank = 2.839`)**, seguido por `B4` (`3.206`), **`DE-S3` (`3.344`)** e `B7` (`3.639`) — todos estatisticamente empatados no bloco vencedor e **estatisticamente superiores** a `DASE` (`5.067`), `B6` (`5.683`), `B3` (`5.700`), `B5` (`6.917`) e `B1` (`8.606`).
2. **Wilcoxon pareado (`DE-S3` nas 6 bases)**:
   - `DE-S3` vs `DASE`: **`+5 / =1 / -0`** (`+0.0073`, 0 derrotas);
   - `DE-S3` vs `B6` e `B3`: **`+4 / =2 / -0`** (0 derrotas);
   - `DE-S3` vs `B5` e `B1`: **`+6 / =0 / -0`**;
   - `DE-S3` vs `B4`: **`+2 / =2 / -2`** (DE vence em `Elec2` e `SEA-Rec`);
   - `DE-S3` vs `GA-S6`: **`+0 / =5 / -1`** (empate em 5 de 6 bases).
3. **Comparação com DETO oficial (5 bases sintéticas)**: tanto `GA-S6` (`0.8765`) quanto `DE-S3` (`0.8738`) superaram o `DETO` (`0.8684`) em desempenho médio, sendo **>1.100× mais rápidos** (`0.36–0.46 s` vs `538.6 s` por stream) e escalando sem falhas para a base real `Elec2`.
4. **Destaque na base real (`Elec2`)**: `DE-S2` (`0.8705`), `GA-S4` (`0.8705`), `DE-S3` (`0.8704`), `DE-S4` (`0.8703`) e `GA-S6` (`0.8700`) superaram todos os baselines e o `DASE` (`0.8669`), com `+0.0139` sobre `B4` (`0.8565`).

---

## 5. Comandos para Reprodução

```powershell
# Testes unitários (27 passed)
pytest -q

# Ablação do DE (21.960 execuções)
python scripts/run_de_ablation.py --seeds 30 --workers 12 --n-min 500

# Ablação do AG (27.360 execuções)
python scripts/run_ga_ablation.py --seeds 30 --workers 12 --n-min 500

# Experimento final otimizado (6 bases, 30 sementes, sem DETO)
python scripts/run_de_deto_dase.py --seeds 30 --workers 12 --n-min 500 --de-strategy current-to-best/1/exp --de-dither --de-CR 0.5 --de-hyper-sigma 0.1 --ga-blx-alpha 0.25 --blocks de ga dase baselines benchmarks detector sensitivity --out results/otimizado_nmin500

# DETO oficial nas 5 bases sintéticas (30 sementes)
python scripts/run_de_deto_dase.py --datasets SEA-Rec STAGGER-Rec AGR-Abr AGR-Grad RBF-Inc --seeds 30 --workers 12 --n-min 500 --blocks deto --out results/deto_nmin500
```
