# Algoritmos Evolucionários Dinâmicos para Ponderação de Ensembles Online sob Concept Drift


## Estrutura de Pastas

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/                  # Configurações YAML da grade experimental (principal, baselines, detector, sensibilidade)
├── data/
│   ├── raw/                  # Bases reais baixadas (INSECTS, Elec2, opcionalmente CreditCard)
│   ├── synthetic/            # Metadados/configurações de streams sintéticas (SEA-Rec, STAGGER-Rec, AGR, RBF-Inc)
│   └── cache/                # Matrizes P (n_instancias x M x C) e rótulos y em .npy para replay rápido
│
├── src/
│   ├── datasets/             # Loaders das bases reais e geradores das streams sintéticas (river)
│   ├── models/               # Pool dos 8 modelos incrementais (river), verificação de diversidade e benchmarks (ARF, SRP, LB)
│   ├── detectors/            # Par de ADWINs, DDM, Oráculo e buffer de warning (B_max=2000, n_min=100)
│   ├── evolution/            # AG de codificação real no simplex Δ^M, reparo, operadores e macro-F1 vetorizado
│   ├── strategies/           # Estratégias pós-drift S1..S6 (reinício, warm start, imigrantes, hipermutação, memória, híbrido)
│   ├── baselines/            # Baselines B1..B8 (uniforme, AG congelado, oráculo, seleção online, desempenho, Hedge, busca aleatória, DE)
│   ├── evaluation/           # Motor de replay prequencial, métricas e testes estatísticos (Friedman/Nemenyi, Wilcoxon+Holm, A12)
│   └── utils/                # Sementes, logging, I/O e utilitários gerais
│
├── scripts/                  # Scripts de entrada (download, diversidade, cache .npy, execução da grade, análise)
├── tests/                    # Testes unitários (reparo no simplex, macro-F1 vs sklearn, detectores, estratégias)
├── notebooks/                # Análise exploratória, piloto (SEA-Rec) e prototipagem de gráficos
├── results/
│   ├── raw/                  # Resultados brutos por execução/semente (CSV)
│   ├── tables/               # Tabelas consolidadas e resultados dos testes estatísticos
│   └── figures/              # Diagramas de diferença crítica, curvas de convergência, macro-F1 e diversidade
└── docs/                     # Relatório, slides e anotações
```

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

No macOS com o Python do python.org, o download do Elec2 pode falhar por certificado SSL;
use `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")`.

## Como reproduzir

O pipeline segue a Seção 10.5 da proposta: o pool roda **uma vez** por base × semente e
todo o resto é replay em numpy sobre as matrizes `P` em cache.

```bash
# 1) Cache das matrizes P (n × M × C), rótulos e predições dos benchmarks (ARF, SRP, LB)
#    EVOCE_CACHE permite guardar o cache fora da pasta sincronizada (padrão: data/cache)
export EVOCE_CACHE=~/evo-ce-data/cache
python scripts/build_cache.py --seeds 30 --workers 8

# 2) Grade experimental: main (S1–S6), baselines (B1–B8), benchmarks, detector, sensibilidade
python scripts/run_experiments.py --seeds 30 --workers 8     # -> results/raw/{runs,events}.csv, curves.npz

# 3) Tabelas, testes estatísticos e figuras
python scripts/analyze.py                                     # -> results/tables, results/figures

# Testes unitários (reparo no simplex, macro-F1 vs sklearn, operadores, estratégias, métricas)
pytest -q
```

| Módulo | Conteúdo |
|---|---|
| `src/evaluation/cache.py` | Geração/leitura do cache `.npz` por base × semente |
| `src/evolution/ga.py` | AG real no simplex (torneio, BLX-α, mutação gaussiana, elitismo), DE/rand/1/bin (B8), busca aleatória (B7) |
| `src/strategies/post_drift.py` | Estratégias S1–S6 e arquivo de memória |
| `src/evaluation/replay.py` | Replay prequencial: detectores (par de ADWINs, DDM, oráculo), buffer do warning, gatilho do AG e baselines B1–B6 |
| `src/evaluation/metrics.py` | Macro-F1, kappa, macro-F1 em janela, recuperação/queda pós-drift, métricas do detector |
| `src/evaluation/stats.py` | Friedman + Nemenyi (diagrama CD), Wilcoxon + Holm, Â12 |

O relatório final está em `docs/relatorio.md` (e `docs/relatorio.pdf`).
