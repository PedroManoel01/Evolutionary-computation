"""Análise dos resultados: tabelas, testes estatísticos e figuras (Seções 10.2–10.3).

Lê results/raw/{runs,events}.csv e curves.npz; escreve results/tables/*.md|csv e results/figures/*.png.

Uso:
    python scripts/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stats import a12, a12_label, cd_diagram, friedman_nemenyi, holm, wilcoxon_p  # noqa: E402

RAW = PROJECT_ROOT / "results" / "raw"
TAB = PROJECT_ROOT / "results" / "tables"
FIG = PROJECT_ROOT / "results" / "figures"
DATASETS = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc", "Elec2"]
STRATS = ["S1", "S2", "S3", "S4", "S5", "S6"]
BASES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
BENCH = ["ARF", "SRP", "LeveragingBagging"]
NAMES = {"S1": "S1 Reinício", "S2": "S2 Warm start", "S3": "S3 WS+imigrantes", "S4": "S4 WS+hipermut.",
         "S5": "S5 Memória", "S6": "S6 Memória+imig.", "B1": "B1 Uniforme", "B2": "B2 AG congelado",
         "B3": "B3 Melhor modelo", "B4": "B4 Seleção online", "B5": "B5 Pond. desempenho",
         "B6": "B6 Hedge", "B7": "B7 Busca aleatória", "B8": "B8 DE", "ARF": "ARF", "SRP": "SRP",
         "LeveragingBagging": "Lev. Bagging"}
DRIFTS = {"SEA-Rec": [6250 * i for i in range(1, 8)], "STAGGER-Rec": [5000 * i for i in range(1, 6)],
          "AGR-Abr": [10000 * i for i in range(1, 5)], "AGR-Grad": [10000 * i for i in range(1, 5)]}
GROUPS = {"Recorrente (SEA, STAGGER)": ["SEA-Rec", "STAGGER-Rec"],
          "Novo (AGR-Abr, AGR-Grad)": ["AGR-Abr", "AGR-Grad"],
          "Incremental (RBF-Inc)": ["RBF-Inc"], "Real (Elec2)": ["Elec2"]}

out_md: List[str] = []


def md(df: pd.DataFrame, title: str, name: str, floatfmt: str = ".4f") -> None:
    df.to_csv(TAB / f"{name}.csv")
    txt = f"### {title}\n\n" + df.to_markdown(floatfmt=floatfmt) + "\n"
    (TAB / f"{name}.md").write_text(txt)
    out_md.append(txt)


def mean_std(runs: pd.DataFrame, metric: str, methods: List[str]) -> pd.DataFrame:
    g = runs[runs.method.isin(methods)].groupby(["dataset", "method"])[metric]
    m, s = g.mean().unstack(), g.std().unstack()
    m = m.reindex(index=DATASETS, columns=methods)
    s = s.reindex(index=DATASETS, columns=methods)
    return m, s


def fmt_ms(m: pd.DataFrame, s: pd.DataFrame, bold_cols: List[str], digits: int = 4) -> pd.DataFrame:
    out = pd.DataFrame(index=m.index, columns=[NAMES.get(c, c) for c in m.columns], dtype=object)
    for d in m.index:
        best = m.loc[d, bold_cols].max()
        for c in m.columns:
            v = f"{m.loc[d, c]:.{digits}f} ± {s.loc[d, c]:.{digits}f}"
            if c in bold_cols and np.isclose(m.loc[d, c], best):
                v = f"**{v}**"
            out.loc[d, NAMES.get(c, c)] = v
    return out


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RAW / "runs.csv")
    events = pd.read_csv(RAW / "events.csv")
    curves = np.load(RAW / "curves.npz")
    summary: Dict[str, object] = {}

    core = runs[runs.block.isin(["main", "baselines", "benchmarks"])].copy()
    seeds = core.seed.nunique()
    summary["n_seeds"] = int(seeds)
    summary["n_runs"] = int(len(runs))

    # ---- Pool: diversidade e modelos individuais ------------------------------------------------
    models = runs[runs.block == "models"]
    pm = models.groupby(["dataset", "method"]).macro_f1.mean().unstack().reindex(DATASETS)
    dis = runs.dropna(subset=["pool_disagreement"]).groupby("dataset").pool_disagreement.mean().reindex(DATASETS)
    pm["Discordância média"] = dis
    md(pm, "Pool: macro-F1 prequencial de cada modelo e discordância média entre pares", "pool")

    # ---- Tabela principal ----------------------------------------------------------------------
    methods = STRATS + BASES
    for metric, label in [("macro_f1", "Macro-F1 prequencial"), ("accuracy", "Acurácia"),
                          ("kappa", "Kappa"), ("wf1_mean", "Macro-F1 médio em janela de 1000")]:
        m, s = mean_std(core, metric, methods + BENCH)
        md(fmt_ms(m, s, methods), f"{label} (média ± desvio, {seeds} sementes). Negrito: melhor entre S1–S6 e B1–B8",
           f"main_{metric}")
        if metric == "macro_f1":
            mean_tab = m

    # ---- Friedman / Nemenyi ----------------------------------------------------------------------
    for key, cols, title in [("all", methods, "S1–S6 e B1–B8"), ("strats", STRATS, "Estratégias S1–S6"),
                             ("withbench", methods + BENCH, "Todos, incluindo benchmarks")]:
        fr = friedman_nemenyi(mean_tab[cols].rename(columns=NAMES))
        cd_diagram(fr["avg_ranks"], fr["cd"], f"{title} — macro-F1, N={fr['N']} bases", str(FIG / f"cd_{key}.png"),
                   fr["p_value"])
        summary[f"friedman_{key}"] = {"p": fr["p_value"], "cd": fr["cd"],
                                      "ranks": fr["avg_ranks"].round(3).to_dict()}
    # complementar: blocos = base x semente (N = 6 x 30)
    piv = core[core.method.isin(STRATS)].pivot_table(index=["dataset", "seed"], columns="method", values="macro_f1")
    fr = friedman_nemenyi(piv[STRATS].rename(columns=NAMES))
    cd_diagram(fr["avg_ranks"], fr["cd"], f"Estratégias S1–S6 — blocos base×semente (N={fr['N']})",
               str(FIG / "cd_strats_seeds.png"), fr["p_value"])
    summary["friedman_strats_seeds"] = {"p": fr["p_value"], "cd": fr["cd"], "ranks": fr["avg_ranks"].round(3).to_dict()}

    ranks_s = summary["friedman_strats"]["ranks"]
    best_s = [k for k, v in NAMES.items() if v == min(ranks_s, key=ranks_s.get)][0]
    summary["best_strategy"] = best_s

    # ---- Wilcoxon + Holm + A12 -------------------------------------------------------------------
    comps = [(s, "S1") for s in STRATS[1:]] + [(best_s, b) for b in ["B7", "B6", "B1", "B5", "B4", "B2", "B8"]] \
        + [("S1", "B7")]
    rows = []
    for d in DATASETS:
        sub = core[core.dataset == d].pivot_table(index="seed", columns="method", values="macro_f1")
        ps = [wilcoxon_p(sub[a], sub[b]) for a, b in comps]
        adj = holm(ps)
        for (a, b), p, pa in zip(comps, ps, adj):
            e = a12(sub[a], sub[b])
            diff = sub[a].mean() - sub[b].mean()
            res = "=" if pa >= 0.05 else ("+" if diff > 0 else "−")
            rows.append({"base": d, "comparação": f"{a} vs {b}", "Δ macro-F1": diff, "p (Holm)": pa,
                         "Â12": e, "efeito": a12_label(e), "resultado": res})
    wt = pd.DataFrame(rows)
    md(wt.set_index(["base", "comparação"]), "Wilcoxon pareado (30 sementes) com correção de Holm por base; "
       "+/− = significativo a 5% a favor/contra o primeiro método", "wilcoxon", ".4g")
    wl = wt.groupby("comparação").resultado.apply(lambda r: f"{(r == '+').sum()}/{(r == '=').sum()}/{(r == '−').sum()}")
    wl = wl.reindex([f"{a} vs {b}" for a, b in comps]).to_frame("vitórias/empates/derrotas (6 bases)")
    md(wl, "Resumo Wilcoxon por comparação", "wilcoxon_summary")

    # ---- H3: convergência e recuperação ------------------------------------------------------------
    main = runs[runs.block == "main"]
    h3 = main.groupby("method")[["init_best", "best_fit", "gens_to_95", "n_gens", "n_evals",
                                 "recovery_time", "max_drop", "div_start", "div_end", "n_events"]].mean()
    h3.index = [NAMES[i] for i in h3.index]
    md(h3, "H3: comportamento evolucionário pós-drift (média sobre bases e sementes; ADWIN)", "h3_overall", ".3f")
    for metric in ["gens_to_95", "init_best", "recovery_time", "max_drop", "n_gens"]:
        t = main.groupby(["dataset", "method"])[metric].mean().unstack().reindex(DATASETS)[STRATS]
        md(t.rename(columns=NAMES), f"H3: {metric} por base", f"h3_{metric}", ".3f")
    # Wilcoxon do tempo de recuperação e das gerações vs S1 (sementes como pares)
    rows = []
    for d in DATASETS:
        sub = main[main.dataset == d]
        for metric in ["recovery_time", "gens_to_95", "n_gens"]:
            pv = sub.pivot_table(index="seed", columns="method", values=metric)
            if any(c not in pv for c in STRATS):
                continue
            pairs = {s: pv[[s, "S1"]].dropna() for s in STRATS[1:]}
            if any(len(p) < 5 for p in pairs.values()):
                continue
            ps = [wilcoxon_p(pairs[s][s], pairs[s]["S1"]) for s in STRATS[1:]]
            adj = holm(ps)
            for s, pa in zip(STRATS[1:], adj):
                diff = pairs[s][s].mean() - pairs[s]["S1"].mean()
                rows.append({"base": d, "métrica": metric, "estratégia": s, "Δ vs S1": diff, "p (Holm)": pa,
                             "Â12": a12(pairs[s][s], pairs[s]["S1"]),
                             "resultado": "=" if pa >= 0.05 else ("menor" if diff < 0 else "maior")})
    if rows:
        md(pd.DataFrame(rows).set_index(["base", "métrica", "estratégia"]),
           "H3: S2–S6 vs S1 em tempo de recuperação e gerações (Wilcoxon + Holm)", "h3_wilcoxon", ".4g")

    # ---- H4: por tipo de drift ----------------------------------------------------------------------
    rows = []
    for g, dss in GROUPS.items():
        sub = main[main.dataset.isin(dss)]
        pv = sub.pivot_table(index=["dataset", "seed"], columns="method", values="macro_f1")[STRATS]
        rk = pv.rank(axis=1, ascending=False).mean()
        rec = sub.groupby("method").recovery_time.mean() if sub.recovery_time.notna().any() else None
        for s in STRATS:
            rows.append({"grupo": g, "estratégia": NAMES[s], "macro-F1": pv[s].mean(), "rank médio": rk[s],
                         "recuperação": rec[s] if rec is not None else np.nan})
    h4 = pd.DataFrame(rows).set_index(["grupo", "estratégia"])
    md(h4, "H4: estratégias por tipo de drift (rank médio sobre base×semente; menor é melhor)", "h4", ".4f")

    # ---- H5: detector --------------------------------------------------------------------------------
    det = runs[(runs.block.isin(["main", "detector"])) & runs.dataset.isin(list(DRIFTS) + ["RBF-Inc"])]
    t = det.groupby(["dataset", "method", "detector"]).macro_f1.mean().unstack()[["adwin", "ddm", "oracle"]]
    t = t.reset_index()
    t["método"] = t.method.map(NAMES)
    md(t.set_index(["dataset", "método"]).drop(columns="method"), "H5: macro-F1 por detector", "h5_f1")
    dm = det.groupby(["dataset", "detector"])[["n_detections", "detection_delay", "false_alarms",
                                               "missed_drifts"]].mean()
    md(dm, "H5: métricas dos detectores (média sobre estratégias e sementes)", "h5_detector", ".1f")
    # decomposição: variação devida ao detector vs à estratégia
    rows = []
    for d in DRIFTS:
        sub = det[det.dataset == d].groupby(["method", "detector"]).macro_f1.mean().unstack()
        rows.append({"base": d,
                     "oráculo − ADWIN (média estr.)": (sub["oracle"] - sub["adwin"]).mean(),
                     "amplitude entre detectores": (sub.max(1) - sub.min(1)).mean(),
                     "amplitude entre estratégias": (sub.max(0) - sub.min(0)).mean()})
    md(pd.DataFrame(rows).set_index("base"), "H5: efeito do detector vs efeito da estratégia (amplitudes de macro-F1)",
       "h5_decomposition")

    # ---- Sensibilidade --------------------------------------------------------------------------------
    sens = runs[runs.block == "sensitivity"].copy()
    if len(sens):
        base = main[main.dataset.isin(sens.dataset.unique())].copy()
        base["factor"], base["value"] = "padrão", "N=30,G=50,n_min=100,imig=0.3,geracional"
        allr = pd.concat([sens, base])
        t = allr.groupby(["factor", "value", "dataset"]).macro_f1.mean().unstack()
        md(t, "Sensibilidade: macro-F1 médio (sobre S1–S6 e 30 sementes)", "sensitivity")
        t2 = allr.groupby(["factor", "value", "method"]).macro_f1.mean().unstack()[STRATS].rename(columns=NAMES)
        md(t2, "Sensibilidade por estratégia (média de SEA-Rec e AGR-Abr)", "sensitivity_by_strategy")
        t3 = allr.groupby(["factor", "value"])[["n_evals", "n_events", "recovery_time"]].mean()
        md(t3, "Sensibilidade: custo e reação", "sensitivity_cost", ".1f")

    # ---- Custo ---------------------------------------------------------------------------------------
    cost = core.groupby("method")[["time_per_inst_ms", "n_evals", "n_events"]].mean()
    cost = cost.reindex(methods + BENCH)
    cost.index = [NAMES[i] for i in cost.index]
    md(cost, "Custo: tempo por instância (ms, inclui o pool) e avaliações de fitness por stream", "cost", ".3f")

    # ---- Figuras ---------------------------------------------------------------------------------------
    plot_curves(curves, best_s)
    plot_convergence(events)
    plot_diversity(events)
    plot_sensitivity(runs, main)
    plot_weights(events)

    (TAB / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    (TAB / "ALL_TABLES.md").write_text("\n".join(out_md))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


def plot_curves(curves, best_s: str) -> None:
    keys = list(curves.keys())
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    show = [("S1", "adwin"), (best_s, "adwin"), ("B1", ""), ("B6", ""), ("B7", "adwin"), ("ARF", "")]
    if best_s == "S1":
        show[1] = ("S2", "adwin")
    for ax, d in zip(axes.flat, DATASETS):
        for meth, det in show:
            arr = [curves[k] for k in keys if k.startswith(f"{d}|{meth}|{det}|")]
            if not arr:
                continue
            mlen = min(len(a) for a in arr)
            m = np.mean([a[:mlen] for a in arr], axis=0)
            x = np.arange(mlen) * 100
            ax.plot(x[10:], m[10:], lw=1.1, label=NAMES.get(meth, meth))
        for p in DRIFTS.get(d, []):
            ax.axvline(p, color="gray", ls="--", lw=0.7)
        ax.set_title(d)
        ax.set_xlabel("instância")
        ax.set_ylabel("macro-F1 (janela 1000)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_over_time.png", dpi=140)
    plt.close(fig)


def plot_convergence(events: pd.DataFrame) -> None:
    ev = events[(events.block == "main") & (~events.initial)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, d in zip(axes.flat, DATASETS):
        for s in STRATS:
            hs = [json.loads(h) for h in ev[(ev.dataset == d) & (ev.method == s)].history]
            if not hs:
                continue
            L = 50
            arr = np.array([h + [h[-1]] * (L - len(h)) for h in hs])[:, :L]
            ax.plot(arr.mean(0), label=NAMES[s], lw=1.3)
        ax.set_title(d)
        ax.set_xlabel("geração")
        ax.set_ylabel("melhor fitness (macro-F1 em E)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Curvas de convergência após o drift (média sobre todos os eventos e sementes)")
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=140)
    plt.close(fig)


def plot_diversity(events: pd.DataFrame) -> None:
    ev = events[(events.block == "main")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, d in zip(axes.flat, DATASETS):
        sub = ev[ev.dataset == d]
        bins = np.linspace(0, sub.t_apply.max() + 1, 11)
        for s in STRATS:
            x = sub[sub.method == s]
            b = np.digitize(x.t_apply, bins)
            m = x.groupby(b).div_start.mean()
            ax.plot(bins[m.index - 1], m.values, marker="o", ms=3, lw=1.2, label=NAMES[s])
        ax.set_title(d)
        ax.set_xlabel("instância do evento")
        ax.set_ylabel("diversidade da pop. inicial")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Diversidade da população no início de cada otimização (distância média entre pares)")
    fig.tight_layout()
    fig.savefig(FIG / "diversity.png", dpi=140)
    plt.close(fig)


def plot_sensitivity(runs: pd.DataFrame, main: pd.DataFrame) -> None:
    sens = runs[runs.block == "sensitivity"]
    if not len(sens):
        return
    factors = ["N", "G", "n_min", "immigrant_rate", "survivor"]
    default = {"N": "30", "G": "50", "n_min": "100", "immigrant_rate": "0.3", "survivor": "generational"}
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, f in zip(axes, factors):
        rows = []
        for d in ["SEA-Rec", "AGR-Abr"]:
            b = main[main.dataset == d].macro_f1.mean()
            rows.append((d, default[f], b))
            for v, g in sens[(sens.factor == f) & (sens.dataset == d)].groupby("value"):
                rows.append((d, v, g.macro_f1.mean()))
        t = pd.DataFrame(rows, columns=["d", "v", "f1"]).pivot(index="v", columns="d", values="f1")
        try:
            t = t.loc[sorted(t.index, key=float)]
        except ValueError:
            pass
        t.plot.bar(ax=ax, rot=0, legend=(f == "N"))
        ax.set_ylim(t.values.min() - 0.01, t.values.max() + 0.005)
        ax.set_title(f)
        ax.set_xlabel("")
    axes[0].set_ylabel("macro-F1 (média S1–S6)")
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png", dpi=140)
    plt.close(fig)


def plot_weights(events: pd.DataFrame) -> None:
    """Pesos escolhidos pelo AG ao longo do stream (seed 0, estratégia S5), por base."""
    names = ["GNB", "HT", "HAT", "EFDT", "Softmax", "KNN", "Bagging", "AMF"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, d in zip(axes.flat, DATASETS):
        sub = events[(events.block == "main") & (events.dataset == d) & (events.method == "S5") & (events.seed == 0)]
        if not len(sub):
            continue
        W = np.array([json.loads(w) for w in sub.weights])
        x = sub.t_apply.values
        ax.stackplot(x, W.T, labels=names, step="post")
        for p in DRIFTS.get(d, []):
            ax.axvline(p, color="k", ls="--", lw=0.7)
        ax.set_title(d)
        ax.set_ylim(0, 1)
        ax.set_xlabel("instância de aplicação")
    axes.flat[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Vetores de pesos evoluídos a cada reotimização (S5, semente 0)")
    fig.tight_layout()
    fig.savefig(FIG / "weights.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
