# Previsão de chuva horária em Recife com ensembles online: uma análise estatística experimental

Projeto da disciplina **IN1119 – Princípios e Técnicas da Análise Estatística Experimental** (CIn/UFPE).

## Problema e base de dados

- **Base real:** dados horários da estação meteorológica automática **A301 – Recife/PE** do
  INMET (Instituto Nacional de Meteorologia), baixados do portal oficial de dados abertos
  <https://portal.inmet.gov.br/dadoshistoricos>. Não vem de repositório de aprendizado de
  máquina (UCI, Kaggle, OpenML); são dados públicos de órgão federal, de uso livre.
- **Tarefa:** a cada hora *t*, com o que foi medido até *t* (pressão, temperatura, ponto de
  orvalho, umidade, radiação, chuva e suas variações), prever **se vai chover na hora t+1**.
- **Por que é um stream com drift:** o regime de chuvas muda ao longo do ano (estação chuvosa
  abr–jul) e entre anos, então a relação atributos → chuva não é estacionária.

## Objetivo

Verificar, com testes estatísticos, se ponderar um ensemble de 8 classificadores online por
um **algoritmo genético** reotimizado a cada drift melhora o **macro-F1** em relação a
alternativas mais simples (média uniforme, seleção do melhor modelo recente, busca aleatória
com o mesmo orçamento) e a um método de referência da literatura (Adaptive Random Forest), e
se o desempenho depende da estação do ano.

## Reprodução

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1) baixe os .zip anuais do INMET e extraia os CSVs da A301 para data/raw/
python scripts/prepare_inmet.py                               # limpeza -> data/processed/
python scripts/build_cache.py --seeds 30 --workers 8          # pool de modelos (test-then-train)
python scripts/run_experiments.py --seeds 30 --workers 8      # 7 métodos x 30 sementes
python scripts/analyze.py                                     # tabelas e figuras
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `data/raw/` | CSVs originais do INMET (um por ano) |
| `data/processed/` | stream limpo `recife_a301.csv` e log da limpeza |
| `src/` | modelos online, AG, estratégias pós-drift, detector ADWIN, métricas |
| `scripts/` | preparação, experimento e análise estatística |
| `results/` | `raw/runs.csv`, `tables/`, `figures/` |
| slides | apresentação de 15 minutos: https://claude.ai/artifact/RhCaPfjSTrH4pyqMbJ5V47 (exporta para PPTX/PDF) |

O núcleo de aprendizado (pool de modelos, AG no simplex, estratégias e replay) foi
reaproveitado do projeto de Computação Evolucionária do mesmo autor; base de dados, problema,
desenho experimental e toda a análise estatística são deste projeto.
