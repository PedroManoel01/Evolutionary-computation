"""Limpa os CSVs horários do INMET (estação automática A301 – Recife) e monta o stream.

Entrada : data/raw/INMET_NE_PE_A301_RECIFE_*.CSV  (um arquivo por ano, baixados do
          portal oficial https://portal.inmet.gov.br/dadoshistoricos)
Saída   : data/processed/recife_a301.csv  (uma linha por hora, ordem cronológica)
          data/processed/limpeza.json     (contagens de cada etapa de limpeza)

Tarefa  : no instante t, usando só o que foi medido até t, prever se vai chover na
          hora seguinte -> alvo = 1 se a precipitação em t+1 > 0 mm.

Uso:
    python scripts/prepare_inmet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# nomes originais (prefixo) -> nomes curtos
COLS = {
    "PRECIPITA": "chuva",
    "PRESSAO ATMOSFERICA AO NIVEL": "pressao",
    "PRESSÃO ATMOSFERICA MAX": "pressao_max",
    "PRESSÃO ATMOSFERICA MIN": "pressao_min",
    "RADIACAO GLOBAL": "radiacao",
    "TEMPERATURA DO AR": "temp",
    "TEMPERATURA DO PONTO DE ORVALHO": "orvalho",
    "TEMPERATURA MÁXIMA": "temp_max",
    "TEMPERATURA MÍNIMA": "temp_min",
    "TEMPERATURA ORVALHO MAX": "orvalho_max",
    "TEMPERATURA ORVALHO MIN": "orvalho_min",
    "UMIDADE REL. MAX": "umid_max",
    "UMIDADE REL. MIN": "umid_min",
    "UMIDADE RELATIVA DO AR": "umid",
    "VENTO, DIRE": "vento_dir",
    "VENTO, RAJADA": "rajada",
    "VENTO, VELOCIDADE": "vento_vel",
}
# vento fica de fora: o anemômetro da A301 falhou em 2019 (42% faltante) e 2020 (90%);
# mantê-lo descartaria quase dois anos inteiros do stream.
EXCLUIR = ["vento_dir", "rajada", "vento_vel"]
MAX_GAP_H = 3  # lacunas de até 3 h são preenchidas pelo último valor; maiores viram NaN


def _rename(c: str) -> str:
    cu = c.upper()
    if cu.startswith("DATA"):
        return "data"
    if cu.startswith("HORA"):
        return "hora"
    for k, v in COLS.items():
        if cu.startswith(k):
            return v
    return c


def read_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", skiprows=8, encoding="latin-1",
                     na_values=["-9999", "-9999,0", ""], dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [_rename(c) for c in df.columns]
    # formatos variam entre anos: "2015-01-01"/"2019/01/01" e "00:00"/"0000 UTC"
    data = df["data"].str.replace("/", "-", regex=False)
    hora = df["hora"].str.replace(" UTC", "", regex=False).str.replace(":", "", regex=False).str.zfill(4)
    df["ts"] = pd.to_datetime(data + " " + hora, format="%Y-%m-%d %H%M")
    for c in COLS.values():
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    return df[["ts", *COLS.values()]]


def main() -> None:
    files = sorted(RAW.glob("INMET_NE_PE_A301_RECIFE_*.CSV"))
    if not files:
        raise SystemExit(f"Nenhum CSV da A301 em {RAW}")
    df = pd.concat([read_year(f) for f in files]).drop_duplicates("ts").sort_values("ts")
    log = {"arquivos": [f.name for f in files], "linhas_brutas": int(len(df))}

    # grade horária completa: horas ausentes no arquivo viram NaN explícito
    df = df.set_index("ts").asfreq("h")
    log["horas_na_grade"] = int(len(df))
    log["faltantes_por_variavel_pct"] = {c: round(float(df[c].isna().mean() * 100), 2) for c in COLS.values()}
    log["faltantes_por_ano_pct"] = {str(a): round(float(g["chuva"].isna().mean() * 100), 1)
                                    for a, g in df.groupby(df.index.year)}
    # anos sem nenhuma medição (estação fora do ar) saem da grade
    vazios = [a for a, g in df.groupby(df.index.year) if g["chuva"].isna().all()]
    df = df[~df.index.year.isin(vazios)]
    log["anos_sem_dados"] = [int(a) for a in vazios]

    # radiação ausente à noite é zero físico (sol abaixo do horizonte; Recife = UTC-3)
    hora_local = (df.index.hour - 3) % 24
    noite = (hora_local < 5) | (hora_local > 18)
    df.loc[noite & df["radiacao"].isna(), "radiacao"] = 0.0
    df.loc[df["radiacao"] < 0, "radiacao"] = 0.0

    # alvo: chuva na PRÓXIMA hora (calculado antes de preencher lacunas -> alvo nunca é inventado)
    df["alvo"] = (df["chuva"].shift(-1) > 0).astype(float)
    df.loc[df["chuva"].shift(-1).isna(), "alvo"] = np.nan

    df = df.drop(columns=EXCLUIR)
    feats = [c for c in COLS.values() if c not in EXCLUIR]
    df[feats] = df[feats].ffill(limit=MAX_GAP_H)

    # atributos derivados (só usam passado e presente)
    df["dp_1h"] = df["pressao"].diff(1)
    df["dp_3h"] = df["pressao"].diff(3)
    df["du_1h"] = df["umid"].diff(1)
    df["chuva_3h"] = df["chuva"].rolling(3, min_periods=3).sum()
    df["hora_sin"] = np.sin(2 * np.pi * hora_local / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora_local / 24)

    antes = len(df)
    df = df.dropna()
    log["linhas_descartadas_por_lacuna"] = int(antes - len(df))
    log["linhas_finais"] = int(len(df))
    log["prevalencia_chuva"] = round(float(df["alvo"].mean()), 4)
    log["periodo"] = [str(df.index.min()), str(df.index.max())]

    OUT.mkdir(parents=True, exist_ok=True)
    df["alvo"] = df["alvo"].astype(int)
    df.reset_index().to_csv(OUT / "recife_a301.csv", index=False)
    (OUT / "limpeza.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
