"""Análise estatística completa: base de dados + experimento.

Parte A – a base (INMET A301 – Recife)
  A1 descritiva e faltantes; A2 chuva x mês (qui-quadrado + V de Cramér);
  A3 variáveis com x sem chuva na hora seguinte (Mann-Whitney + r de postos).
Parte B – o experimento (7 métodos x 30 sementes)
  B1 descritiva com IC95% t e bootstrap; B2 pressupostos (Shapiro-Wilk, Levene);
  B3 ANOVA de medidas repetidas + Friedman; B4 comparações pareadas planejadas
  (t pareado e Wilcoxon, Holm, d_z de Cohen, Â12); B5 Friedman + Nemenyi com os
  anos como blocos; B6 ANOVA de dois fatores método x estação; B7 análise de poder.

Saídas: results/tables/*.csv|md, results/figures/*.png, results/tables/resumo.json

Uso:
    python scripts/analyze.py
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.anova import AnovaRM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.real import load_frame  # noqa: E402
from src.evaluation.stats import a12, a12_label, cd_diagram, friedman_nemenyi, holm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
ALPHA = 0.05
NOMES = {"B1": "Média uniforme", "B4": "Seleção online", "B7": "Busca aleatória",
         "S1": "AG reinício", "S2": "AG warm start", "S6": "AG memória+imigr.", "ARF": "ARF"}
ORDEM = list(NOMES)
# comparações planejadas (hipóteses do trabalho)
PLANEJADAS = [("S2", "B1"), ("S2", "B4"), ("S2", "B7"), ("S2", "ARF"), ("S2", "S1"), ("S6", "S1")]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
resumo: dict = {}


def save(df: pd.DataFrame, name: str, title: str, index: bool = True) -> None:
    df.to_csv(TAB / f"{name}.csv", index=index)
    (TAB / f"{name}.md").write_text(f"### {title}\n\n{df.to_markdown(index=index, floatfmt='.4g')}\n")


def boot_ci(x: np.ndarray, B: int = 10000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cramers_v(tab: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


# ----------------------------------------------------------------------------- Parte A
def parte_a() -> None:
    df = load_frame()
    limp = json.loads((ROOT / "data" / "processed" / "limpeza.json").read_text())
    df["mes"] = df["ts"].dt.month
    df["ano"] = df["ts"].dt.year
    resumo["base"] = {"n": len(df), "periodo": limp["periodo"], "prevalencia": limp["prevalencia_chuva"],
                      "descartadas": limp["linhas_descartadas_por_lacuna"], "horas_grade": limp["horas_na_grade"]}

    # A1 descritiva
    vars_ = ["temp", "umid", "pressao", "orvalho", "radiacao", "chuva"]
    d = df[vars_].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    d.columns = ["média", "dp", "mín", "Q1", "mediana", "Q3", "máx"]
    save(d, "a1_descritiva", "Estatística descritiva das variáveis (horas válidas)")
    falt = pd.Series(limp["faltantes_por_variavel_pct"], name="% faltante na grade horária").to_frame()
    save(falt, "a1_faltantes", "Percentual de horas sem medição por variável (antes da limpeza)")
    por_ano = df.groupby("ano").agg(horas=("alvo", "size"), prev_chuva=("alvo", "mean"))
    save(por_ano, "a1_por_ano", "Horas válidas e prevalência de chuva por ano")

    # A2 chuva x mês
    tab = pd.crosstab(df["mes"], df["alvo"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = cramers_v(tab)
    resumo["A2_qui2_mes"] = {"chi2": chi2, "gl": dof, "p": p, "V_cramer": v}
    prev = df.groupby("mes")["alvo"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(MESES, prev.values * 100, color=["#3b82f6" if 4 <= m <= 7 else "#94a3b8" for m in prev.index])
    ax.set_ylabel("% de horas com chuva na hora seguinte")
    ax.set_title(f"Prevalência de chuva por mês (χ² = {chi2:.0f}, gl = {dof}, p < 0,001, V = {v:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "a2_chuva_por_mes.png", dpi=150); plt.close(fig)

    # A3 variáveis com x sem chuva
    rows = []
    for c in ["umid", "pressao", "temp", "radiacao", "dp_3h", "chuva"]:
        a, b = df.loc[df.alvo == 1, c], df.loc[df.alvo == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"variável": c, "mediana chuva": a.median(), "mediana sem chuva": b.median(),
                     "U": u, "p": p, "r rank-bisserial": 2 * u / (len(a) * len(b)) - 1})
    save(pd.DataFrame(rows), "a3_mann_whitney", "Mann-Whitney: horas seguidas de chuva x sem chuva", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, c, lab in zip(axs, ["umid", "pressao", "temp"], ["Umidade (%)", "Pressão (mB)", "Temperatura (°C)"]):
        ax.boxplot([df.loc[df.alvo == 0, c], df.loc[df.alvo == 1, c]], showfliers=False)
        ax.set_xticks([1, 2], ["sem chuva", "chuva"]); ax.set_title(lab)
    fig.suptitle("Condições na hora t conforme chover ou não em t+1")
    fig.tight_layout(); fig.savefig(FIG / "a3_boxplots.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- Parte B
def pivot(runs: pd.DataFrame, bloco: str = "global", nivel: str = "global") -> pd.DataFrame:
    r = runs[(runs.bloco == bloco) & (runs.nivel == nivel)]
    return r.pivot(index="seed", columns="metodo", values="macro_f1")[ORDEM]


def parte_b() -> None:
    runs = pd.read_csv(ROOT / "results" / "raw" / "runs.csv", dtype={"nivel": str})
    W = pivot(runs)
    n = len(W)

    # B1 descritiva
    rows = []
    for m in ORDEM:
        x = W[m].values
        t_lo, t_hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x)) if x.std() > 0 else (x.mean(),) * 2
        b_lo, b_hi = boot_ci(x)
        rows.append({"método": f"{m} – {NOMES[m]}", "média": x.mean(), "dp": x.std(ddof=1), "mediana": np.median(x),
                     "IC95% t": f"[{t_lo:.4f}; {t_hi:.4f}]", "IC95% bootstrap": f"[{b_lo:.4f}; {b_hi:.4f}]"})
    save(pd.DataFrame(rows), "b1_descritiva", f"Macro-F1 global por método ({n} sementes)", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.boxplot([W[m] for m in ORDEM], showmeans=True)
    ax.set_xticks(range(1, len(ORDEM) + 1), [f"{m}\n{NOMES[m]}" for m in ORDEM], fontsize=8)
    ax.set_ylabel("Macro-F1"); ax.set_title(f"Macro-F1 prequencial por método ({n} sementes)")
    fig.tight_layout(); fig.savefig(FIG / "b1_boxplot.png", dpi=150); plt.close(fig)

    # B2 pressupostos
    rows = []
    for m in ORDEM:
        x = W[m].values
        p = stats.shapiro(x).pvalue if x.std() > 0 else np.nan
        rows.append({"amostra": m, "W de Shapiro p": p, "normal (5%)": "sim" if p > ALPHA else "não"})
    for a, b in PLANEJADAS:
        d = (W[a] - W[b]).values
        p = stats.shapiro(d).pvalue if d.std() > 0 else np.nan
        rows.append({"amostra": f"{a} − {b}", "W de Shapiro p": p, "normal (5%)": "sim" if p > ALPHA else "não"})
    lev = stats.levene(*[W[m] for m in ORDEM], center="median")
    resumo["B2_levene"] = {"W": lev.statistic, "p": lev.pvalue}
    save(pd.DataFrame(rows), "b2_normalidade", "Shapiro-Wilk por método e nas diferenças pareadas", index=False)
    fig, axs = plt.subplots(1, len(PLANEJADAS), figsize=(14, 2.8))
    for ax, (a, b) in zip(axs, PLANEJADAS):
        stats.probplot((W[a] - W[b]).values, plot=ax)
        ax.set_title(f"QQ {a} − {b}", fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG / "b2_qqplots.png", dpi=150); plt.close(fig)

    # B3 omnibus
    long = W.reset_index().melt(id_vars="seed", var_name="metodo", value_name="f1")
    rm = AnovaRM(long, "f1", "seed", within=["metodo"]).fit().anova_table
    fr = stats.friedmanchisquare(*[W[m] for m in ORDEM])
    k = len(ORDEM)
    kendall_w = fr.statistic / (n * (k - 1))
    resumo["B3_anova_rm"] = {"F": float(rm["F Value"].iloc[0]), "gl1": float(rm["Num DF"].iloc[0]),
                             "gl2": float(rm["Den DF"].iloc[0]), "p": float(rm["Pr > F"].iloc[0])}
    resumo["B3_friedman"] = {"chi2": fr.statistic, "p": fr.pvalue, "W_kendall": kendall_w}

    # B4 comparações planejadas
    rows = []
    for a, b in PLANEJADAS:
        d = (W[a] - W[b]).values
        sd = d.std(ddof=1)
        t = stats.ttest_rel(W[a], W[b]) if sd > 0 else None
        w = stats.wilcoxon(W[a], W[b], zero_method="zsplit") if np.any(d != 0) else None
        rows.append({"comparação": f"{a} vs {b}", "Δ médio": d.mean(),
                     "IC95% Δ": "[{:.4f}; {:.4f}]".format(*stats.t.interval(0.95, n - 1, d.mean(), stats.sem(d))) if sd > 0 else "-",
                     "t": t.statistic if t else np.nan, "p t": t.pvalue if t else 1.0,
                     "p Wilcoxon": w.pvalue if w else 1.0, "d_z": d.mean() / sd if sd > 0 else 0.0,
                     "Â12": a12(W[a], W[b]), "efeito Â12": a12_label(a12(W[a], W[b]))})
    B4 = pd.DataFrame(rows)
    B4["p t (Holm)"] = holm(B4["p t"])
    B4["p Wilcoxon (Holm)"] = holm(B4["p Wilcoxon"])
    B4["conclusão"] = np.where(B4["p Wilcoxon (Holm)"] < ALPHA,
                               np.where(B4["Δ médio"] > 0, "1º melhor", "1º pior"), "sem diferença")
    save(B4, "b4_planejadas", "Comparações planejadas (t pareado e Wilcoxon, correção de Holm)", index=False)
    # todas as pares (Wilcoxon + Holm), para referência
    rows = []
    for a, b in combinations(ORDEM, 2):
        d = (W[a] - W[b]).values
        rows.append({"par": f"{a} vs {b}", "Δ": d.mean(),
                     "p": stats.wilcoxon(W[a], W[b], zero_method="zsplit").pvalue if np.any(d != 0) else 1.0})
    allp = pd.DataFrame(rows)
    allp["p Holm"] = holm(allp["p"])
    save(allp, "b4_todos_pares", "Wilcoxon pareado entre todos os pares (Holm)", index=False)

    # B5 anos como blocos
    Y = runs[runs.bloco == "ano"].groupby(["nivel", "metodo"])["macro_f1"].mean().unstack()[ORDEM]
    fn = friedman_nemenyi(Y)
    resumo["B5_friedman_anos"] = {"p": fn["p_value"], "CD": fn["cd"], "N_anos": fn["N"],
                                  "ranks": fn["avg_ranks"].round(3).to_dict()}
    save(Y, "b5_f1_por_ano", "Macro-F1 médio (30 sementes) por ano")
    cd_diagram(fn["avg_ranks"], fn["cd"], "Ranks médios com os anos como blocos (Nemenyi, α = 0,05)",
               str(FIG / "b5_cd_anos.png"), p_value=fn["p_value"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in ORDEM:
        ax.plot(Y.index, Y[m], marker="o", ms=3, label=m)
    ax.set_ylabel("Macro-F1"); ax.set_title("Macro-F1 por ano"); ax.legend(ncol=7, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "b5_f1_por_ano.png", dpi=150); plt.close(fig)

    # B6 método x estação: ANOVA de medidas repetidas de dois fatores (a mesma semente é
    # medida nas duas estações e em todos os métodos -> sujeito = semente)
    E = runs[runs.bloco == "estacao"][["seed", "metodo", "nivel", "macro_f1"]].rename(columns={"nivel": "estacao"})
    av = AnovaRM(E, "macro_f1", "seed", within=["metodo", "estacao"]).fit().anova_table
    av.columns = ["F", "gl1", "gl2", "p"]
    save(av, "b6_anova_estacao", "ANOVA de medidas repetidas (sujeito = semente): macro-F1 ~ método × estação")
    resumo["B6_anova_rm2"] = {i: {"F": float(r.F), "p": float(r.p)} for i, r in av.iterrows()}
    # robusto: a queda chuvosa -> seca difere entre métodos? (Friedman na queda = interação)
    queda = (E[E.estacao == "chuvosa"].pivot(index="seed", columns="metodo", values="macro_f1")
             - E[E.estacao == "seca"].pivot(index="seed", columns="metodo", values="macro_f1"))[ORDEM]
    frq = stats.friedmanchisquare(*[queda[m] for m in ORDEM])
    resumo["B6_friedman_queda"] = {"chi2": frq.statistic, "p": frq.pvalue,
                                   "queda_media": queda.mean().round(4).to_dict()}
    save(queda.describe().T[["mean", "std"]].rename(columns={"mean": "queda média", "std": "dp"}),
         "b6_queda_seca", "Queda do macro-F1 da estação chuvosa para a seca, por método")
    resumo["B6_friedman_por_estacao"] = {
        est: float(stats.friedmanchisquare(*[pivot(runs, "estacao", est)[m] for m in ORDEM]).pvalue)
        for est in ("chuvosa", "seca")}
    mi = E.groupby(["estacao", "metodo"])["macro_f1"].mean().unstack()[ORDEM]
    save(mi, "b6_medias_estacao", "Macro-F1 médio por método e estação")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m in ORDEM:
        ax.plot(["chuvosa", "seca"], mi.loc[["chuvosa", "seca"], m], marker="o", label=m)
    ax.set_ylabel("Macro-F1"); ax.set_title("Interação método × estação"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "b6_interacao.png", dpi=150); plt.close(fig)

    # B7 poder do t pareado bilateral pela t não central (nc = d_z * sqrt(n))
    def poder(dz: float, nn: int) -> float:
        tc = stats.t.isf(ALPHA / 2, nn - 1)
        nc = dz * np.sqrt(nn)
        if nc > 12:  # a t não central da scipy perde precisão aqui; o poder já é 1 na prática
            return 1.0
        return float(stats.nct.sf(tc, nn - 1, nc) + stats.nct.cdf(-tc, nn - 1, nc))

    rows = []
    for a, b in PLANEJADAS:
        d = (W[a] - W[b]).values
        dz = abs(d.mean()) / d.std(ddof=1) if d.std() > 0 else 0.0
        n80 = next((m for m in range(3, 5001) if poder(dz, m) >= 0.8), np.nan) if dz > 0 else np.nan
        rows.append({"comparação": f"{a} vs {b}", "|d_z|": dz, "poder com n=30": poder(dz, n), "n p/ poder 80%": n80})
    save(pd.DataFrame(rows), "b7_poder", "Análise de poder do t pareado (α = 0,05, bilateral)", index=False)
    resumo["B_n_sementes"] = n


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    parte_a()
    if (ROOT / "results" / "raw" / "runs.csv").exists():
        parte_b()
    (TAB / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(resumo, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
