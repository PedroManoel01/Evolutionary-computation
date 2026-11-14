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
.venv\Scripts\activate
pip install -r requirements.txt
```
