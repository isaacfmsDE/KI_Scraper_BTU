# 4_Validator.py  (Scraper_V4) - CORRIGIDO
# - Sempre roda coverage + sanity + missed
# - SEMPRE roda LLM audit em modo "suspicious"
# - Normaliza datas para ISO antes do LLM (dd/mm/yyyy e dd.mm.yyyy)
# - Evita "MISMATCH" sem evidência: reclassifica para "UNCERTAIN"
# Outputs: tudo dentro da pasta ./Validation

import os
import re
import json
import time
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ==============================
# CONFIG DEFAULTS
# ==============================
MODEL = "gpt-4.1-mini"

DEFAULT_SCRAPER_CSV = "foerderprogramme_286.csv"
DEFAULT_LABELED_CSV = "foerderprogramme_Labeled.csv"
DEFAULT_FEATURES_CSV = "richtlinien_features.csv"
DEFAULT_TXT_DIR = "richtlinien_txt"
DEFAULT_OUT_DIR = "Validation"


# ==============================
# UTILS
# ==============================
def ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


def as_clean_str(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def is_empty_or_nan(v) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    return (not s) or (s.lower() == "nan")


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return None
        s = s.replace("€", "").replace("EUR", "").replace(" ", "")
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        return None


def similarity(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY não encontrada. Verifique .env ou variáveis de ambiente.")
    return OpenAI(api_key=api_key)


# ==============================
# DATE NORMALIZATION (NEW)
# ==============================
def normalize_date_any_to_iso(s: str) -> str:
    """
    Converte formatos comuns para ISO:
    - YYYY-MM-DD (mantém)
    - DD.MM.YYYY
    - DD/MM/YYYY
    Retorna "" se não conseguir.
    """
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if not t:
        return ""

    # ISO já
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t

    # DD.MM.YYYY
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", t)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # DD/MM/YYYY
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    return ""


# ==============================
# LAYER 1: COVERAGE / ALIGNMENT
# ==============================
def coverage_report(
    scraper_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    txt_dir: Path,
    out_dir: Path
) -> Path:
    txt_files = sorted([p.name for p in txt_dir.glob("*.txt")]) if txt_dir.exists() else []
    txt_set = set(txt_files)

    feature_files = []
    if "source_file" in features_df.columns:
        feature_files = [str(x) for x in features_df["source_file"].fillna("").tolist() if str(x).strip()]
    feature_set = set(feature_files)

    rows = []

    for f in sorted(txt_set - feature_set):
        rows.append({
            "type": "TXT_SEM_FEATURE",
            "item": f,
            "detail": "Existe em richtlinien_txt, mas não aparece em richtlinien_features.csv (source_file)."
        })

    for f in sorted(feature_set - txt_set):
        rows.append({
            "type": "FEATURE_SEM_TXT",
            "item": f,
            "detail": "Aparece em richtlinien_features.csv (source_file), mas o .txt não existe em richtlinien_txt."
        })

    rows.append({"type": "STATS", "item": "scraper_rows", "detail": int(scraper_df.shape[0])})
    rows.append({"type": "STATS", "item": "labeled_rows", "detail": int(labeled_df.shape[0])})
    rows.append({"type": "STATS", "item": "features_rows", "detail": int(features_df.shape[0])})
    rows.append({"type": "STATS", "item": "txt_files", "detail": int(len(txt_files))})

    out_path = out_dir / "coverage_report.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ==============================
# LAYER 2: SANITY CHECKS
# ==============================
def sanity_checks(features_df: pd.DataFrame, out_dir: Path) -> Path:
    required_cols = [
        "source_file",
        "titel_der_foerderung",
        "foerdergeber",
        "foerderbereich",
        "foerdersumme",
        "dauer_monate",
        "laufzeit_programm_start",
        "laufzeit_programm_ende",
        "antragsfrist_start",
        "antragsfrist_ende",
    ]
    for c in required_cols:
        if c not in features_df.columns:
            features_df[c] = ""

    flags = []

    for _, row in features_df.iterrows():
        src = str(row.get("source_file", "")).strip()
        titel = str(row.get("titel_der_foerderung", "")).strip()
        geber = str(row.get("foerdergeber", "")).strip()

        summe = safe_float(row.get("foerdersumme"))
        dauer = safe_float(row.get("dauer_monate"))

        # (CHANGED) aceita dd/mm e dd.mm e converte para ISO
        lps = normalize_date_any_to_iso(as_clean_str(row.get("laufzeit_programm_start", "")))
        lpe = normalize_date_any_to_iso(as_clean_str(row.get("laufzeit_programm_ende", "")))
        afs = normalize_date_any_to_iso(as_clean_str(row.get("antragsfrist_start", "")))
        afe = normalize_date_any_to_iso(as_clean_str(row.get("antragsfrist_ende", "")))

        if not titel:
            flags.append({"source_file": src, "flag": "EMPTY_TITEL", "detail": "titel_der_foerderung está vazio."})
        if not geber:
            flags.append({"source_file": src, "flag": "EMPTY_FOERDERGEBER", "detail": "foerdergeber está vazio."})

        if summe is not None:
            if summe < 0:
                flags.append({"source_file": src, "flag": "NEGATIVE_FOERDERSUMME", "detail": f"foerdersumme={summe}"})
            if summe > 1e11:
                flags.append({"source_file": src, "flag": "HUGE_FOERDERSUMME", "detail": f"foerdersumme={summe} (muito alto)"})

        if dauer is not None:
            if dauer < 0:
                flags.append({"source_file": src, "flag": "NEGATIVE_DAUER", "detail": f"dauer_monate={dauer}"})
            if dauer > 120:
                flags.append({"source_file": src, "flag": "VERY_LONG_DAUER", "detail": f"dauer_monate={dauer} (>120)"})

        raw_dates = {
            "laufzeit_programm_start": as_clean_str(row.get("laufzeit_programm_start", "")),
            "laufzeit_programm_ende": as_clean_str(row.get("laufzeit_programm_ende", "")),
            "antragsfrist_start": as_clean_str(row.get("antragsfrist_start", "")),
            "antragsfrist_ende": as_clean_str(row.get("antragsfrist_ende", "")),
        }
        parsed_dates = {
            "laufzeit_programm_start": lps,
            "laufzeit_programm_ende": lpe,
            "antragsfrist_start": afs,
            "antragsfrist_ende": afe,
        }

        for k in raw_dates:
            if raw_dates[k] and not parsed_dates[k]:
                flags.append({"source_file": src, "flag": "BAD_DATE_FORMAT", "detail": f"{k}='{raw_dates[k]}' (não reconhecido)"})

        if lps and lpe and lps > lpe:
            flags.append({"source_file": src, "flag": "LAUFZEIT_INVERTED", "detail": f"{lps} > {lpe}"})
        if afs and afe and afs > afe:
            flags.append({"source_file": src, "flag": "ANTRAGSFRIST_INVERTED", "detail": f"{afs} > {afe}"})

    out_path = out_dir / "sanity_flags.csv"
    pd.DataFrame(flags).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ==============================
# MISSED CANDIDATES
# ==============================
def missed_candidates(labeled_df: pd.DataFrame, out_dir: Path, top_n: int = 40) -> Path:
    for c in ["Titel", "Förderbereich", "Förderart", "Fördergeber", "Förderberechtigte", "Label", "Begruendung"]:
        if c not in labeled_df.columns:
            labeled_df[c] = ""

    def row_text(r):
        return " | ".join([
            str(r.get("Titel", "") or ""),
            str(r.get("Förderbereich", "") or ""),
            str(r.get("Förderart", "") or ""),
            str(r.get("Fördergeber", "") or ""),
            str(r.get("Förderberechtigte", "") or ""),
        ]).strip()

    ja = labeled_df[labeled_df["Label"].astype(str).str.upper() == "JA"].copy()
    nein = labeled_df[labeled_df["Label"].astype(str).str.upper() == "NEIN"].copy()

    ja_texts = [row_text(r) for _, r in ja.iterrows()]

    rows = []
    for _, r in nein.iterrows():
        t_nein = row_text(r)
        best = 0.0
        best_match = ""
        for t_ja in ja_texts:
            s = similarity(t_nein, t_ja)
            if s > best:
                best = s
                best_match = t_ja

        rows.append({
            "Titel": r.get("Titel", ""),
            "Förderbereich": r.get("Förderbereich", ""),
            "Förderart": r.get("Förderart", ""),
            "Fördergeber": r.get("Fördergeber", ""),
            "Förderberechtigte": r.get("Förderberechtigte", ""),
            "Label": r.get("Label", ""),
            "Begruendung": r.get("Begruendung", ""),
            "similarity_to_best_JA": round(best, 3),
            "best_JA_signature": best_match[:300],
        })

    out_df = pd.DataFrame(rows).sort_values("similarity_to_best_JA", ascending=False).head(top_n)
    out_path = out_dir / "missed_candidates.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ==============================
# LLM AUDIT (sempre "suspicious")
# ==============================
AUDIT_PROMPT = """
Du bist ein strenger Prüfer. Du bekommst:
(1) extrahierte Werte (CSV)
(2) Originaltext (Richtlinie)

Prüfe NUR diese Felder:
- laufzeit_programm_start, laufzeit_programm_ende
- antragsfrist_start, antragsfrist_ende
- foerdersumme
- dauer_monate

WICHTIGE REGEL:
Wenn ein Datum, eine Fördersumme oder eine Laufzeit NICHT explizit im Text steht,
darfst du KEINEN Wert raten oder ableiten.
In diesem Fall MUSS das Feld leer ("") oder 0 sein.
Wenn im Text nur steht "läuft bis auf Weiteres" oder "gültig ab Veröffentlichung",
darf KEIN Datum erzeugt werden.

SPEZIALREGEL LAUFZEIT:
- Wenn im Text ein primäres Ende als "befristet bis <DATUM>" / "mithin bis <DATUM>" steht,
  ist dieses Datum das laufzeit_programm_ende.
- Formulierungen wie "nicht über den <DATUM> hinaus" sind ein MAXIMALER Deckel (bedingt)
  und ersetzen NICHT das primäre Ende.

KLASSIFIKATION:
- "CONFIRMED": CSV-Wert stimmt mit Text überein (oder sinnvoll äquivalent).
- "MISSING_IN_TEXT": Kein expliziter Wert im Text; leer/NaN im CSV ist plausibel.
- "MISSING_IN_CSV": Wert ist im Text explizit, aber CSV ist leer/NaN/0.
- "MISMATCH": Text enthält einen expliziten Wert, aber CSV weicht ab.
  WICHTIG: Nutze "MISMATCH" nur, wenn du im Text einen expliziten Gegenwert findest.

EVIDENCE:
- Wenn Text einen Wert enthält, gib eine kurze Textstelle (max 20 Wörter).
- Wenn kein Wert im Text, evidence="".

Antwortformat: NUR gültiges JSON:

{
  "source_file": "",
  "checks": [
    {
      "field": "antragsfrist_start",
      "csv_value": "",
      "found_value": "",
      "classification": "CONFIRMED",
      "evidence": ""
    }
  ],
  "overall": "OK" | "SUSPECT"
}

Regeln:
- Keine zusätzlichen Felder.
- Kein Text außerhalb JSON.
""".strip()


DATE_FIELDS = [
    "laufzeit_programm_start",
    "laufzeit_programm_ende",
    "antragsfrist_start",
    "antragsfrist_ende",
]


def _postprocess_audit_json(data: dict) -> dict:
    """
    (NEW) Corrige casos perigosos para o Cleaner:
    - Se vier MISMATCH sem found_value e sem evidence -> reclassifica para UNCERTAIN
      (Cleaner ignora e não apaga nada).
    """
    if not isinstance(data, dict):
        return {"source_file": "", "checks": [], "overall": "SUSPECT"}

    checks = data.get("checks", []) or []
    overall = data.get("overall", "SUSPECT")

    fixed = []
    for c in checks:
        if not isinstance(c, dict):
            continue

        cls = (c.get("classification", "") or "").strip()
        found = (c.get("found_value", "") or "").strip()
        ev = (c.get("evidence", "") or "").strip()

        # Proteção principal:
        if cls == "MISMATCH" and (not found) and (not ev):
            c["classification"] = "UNCERTAIN"
            overall = "SUSPECT"

        fixed.append(c)

    data["checks"] = fixed
    data["overall"] = overall if overall in ("OK", "SUSPECT") else "SUSPECT"
    return data




def llm_audit_suspicious(
    client: OpenAI,
    features_df: pd.DataFrame,
    txt_dir: Path,
    out_dir: Path,
    sleep_s: float = 0.8
) -> Path:
    out_path = out_dir / "audit_llm.jsonl"

    needed = [
        "source_file",
        "titel_der_foerderung",
        "foerdergeber",
        "foerdersumme",
        "dauer_monate",
        "laufzeit_programm_start",
        "laufzeit_programm_ende",
        "antragsfrist_start",
        "antragsfrist_ende",
    ]
    for c in needed:
        if c not in features_df.columns:
            features_df[c] = ""

    candidates = []
    for _, r in features_df.iterrows():
        src = str(r.get("source_file", "")).strip()
        if not src:
            continue

        titel = str(r.get("titel_der_foerderung", "")).strip()
        geber = str(r.get("foerdergeber", "")).strip()
        summe = safe_float(r.get("foerdersumme"))
        dauer_missing = is_empty_or_nan(r.get("dauer_monate"))

        # (CHANGED) datas: considera "suspeito" se está vazio OU se não dá para normalizar para ISO
        date_suspect = False
        for f in DATE_FIELDS:
            raw = as_clean_str(r.get(f, ""))
            if not raw:
                date_suspect = True
                break
            if not normalize_date_any_to_iso(raw):
                date_suspect = True
                break

        if (not titel) or (not geber) or (summe is None) or (summe == 0) or dauer_missing or date_suspect:
            candidates.append(src)

    candidates = sorted(set(candidates))

    with open(out_path, "w", encoding="utf-8") as f:
        for src in candidates:
            txt_path = txt_dir / src
            if not txt_path.exists():
                continue

            row = features_df[features_df["source_file"].astype(str) == src]
            if row.empty:
                continue
            row = row.iloc[0].to_dict()

            # (CHANGED) manda datas em ISO para o LLM (reduz falso mismatch)
            payload = {
                "source_file": src,
                "foerdersumme": as_clean_str(row.get("foerdersumme", "")),
                "dauer_monate": as_clean_str(row.get("dauer_monate", "")),
                "laufzeit_programm_start": normalize_date_any_to_iso(as_clean_str(row.get("laufzeit_programm_start", ""))),
                "laufzeit_programm_ende": normalize_date_any_to_iso(as_clean_str(row.get("laufzeit_programm_ende", ""))),
                "antragsfrist_start": normalize_date_any_to_iso(as_clean_str(row.get("antragsfrist_start", ""))),
                "antragsfrist_ende": normalize_date_any_to_iso(as_clean_str(row.get("antragsfrist_ende", ""))),
            }

            text = txt_path.read_text(encoding="utf-8-sig", errors="ignore")
            user_msg = (
                AUDIT_PROMPT
                + "\n\nEXTRACTED_VALUES:\n"
                + json.dumps(payload, ensure_ascii=False)
                + "\n\nRICHTLINIE_TEXT:\n"
                + '"""\n' + text + '\n"""\n'
            )

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0
            )

            raw = (resp.choices[0].message.content or "").strip()

            try:
                data = json.loads(raw)
            except Exception:
                data = {"source_file": src, "checks": [], "overall": "SUSPECT"}

            # (NEW) pós-processamento para evitar CLEAR_MISMATCH indevido
            data = _postprocess_audit_json(data)

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            time.sleep(sleep_s)

    return out_path


def audit_jsonl_to_csv(jsonl_path: Path, out_csv: Path) -> Path:
    rows = []
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out_csv

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            source = obj.get("source_file", "")
            overall = obj.get("overall", "")
            checks = obj.get("checks", []) or []
            for c in checks:
                rows.append({
                    "source_file": source,
                    "field": c.get("field", ""),
                    "csv_value": c.get("csv_value", ""),
                    "found_value": c.get("found_value", ""),
                    "classification": c.get("classification", ""),
                    "evidence": c.get("evidence", ""),
                    "overall": overall,
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


# ==============================
# PUBLIC ENTRYPOINT (para o pipeline)
# ==============================
def run_validator(
    scraper_csv: str = DEFAULT_SCRAPER_CSV,
    labeled_csv: str = DEFAULT_LABELED_CSV,
    features_csv: str = DEFAULT_FEATURES_CSV,
    txt_dir: str = DEFAULT_TXT_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    audit_sleep: float = 0.8,
    missed_top_n: int = 40
) -> dict:
    base = Path(__file__).resolve().parent
    scraper_path = (base / scraper_csv).resolve()
    labeled_path = (base / labeled_csv).resolve()
    features_path = (base / features_csv).resolve()
    txt_dir_path = (base / txt_dir).resolve()
    out_dir_path = (base / out_dir).resolve()

    ensure_outdir(out_dir_path)

    if not scraper_path.exists():
        raise FileNotFoundError(f"Não encontrei: {scraper_path}")
    if not labeled_path.exists():
        raise FileNotFoundError(f"Não encontrei: {labeled_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Não encontrei: {features_path}")
    if not txt_dir_path.exists():
        raise FileNotFoundError(f"Não encontrei: {txt_dir_path}")

    scraper_df = read_csv_robust(scraper_path)
    labeled_df = read_csv_robust(labeled_path)
    features_df = read_csv_robust(features_path)

    cov = coverage_report(scraper_df, labeled_df, features_df, txt_dir_path, out_dir_path)
    san = sanity_checks(features_df, out_dir_path)
    miss = missed_candidates(labeled_df, out_dir_path, top_n=missed_top_n)

    client = build_client()
    audit_jsonl = llm_audit_suspicious(
        client=client,
        features_df=features_df,
        txt_dir=txt_dir_path,
        out_dir=out_dir_path,
        sleep_s=audit_sleep
    )

    audit_csv = audit_jsonl_to_csv(audit_jsonl, out_dir_path / "audit_llm.csv")

    return {
        "coverage_report": str(cov),
        "sanity_flags": str(san),
        "missed_candidates": str(miss),
        "audit_jsonl": str(audit_jsonl),
        "audit_csv": str(audit_csv),
        "out_dir": str(out_dir_path),
    }


# ==============================
# CLI
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Validator (auto suspicious audit + jsonl->csv)")
    parser.add_argument("--scraper", default=DEFAULT_SCRAPER_CSV)
    parser.add_argument("--labeled", default=DEFAULT_LABELED_CSV)
    parser.add_argument("--features", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--txt-dir", default=DEFAULT_TXT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-sleep", type=float, default=0.8)
    parser.add_argument("--missed-top-n", type=int, default=40)
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    out = run_validator(
        scraper_csv=args.scraper,
        labeled_csv=args.labeled,
        features_csv=args.features,
        txt_dir=args.txt_dir,
        out_dir=args.out_dir,
        audit_sleep=args.audit_sleep,
        missed_top_n=args.missed_top_n
    )

    print(f"[OK] Coverage report: {out['coverage_report']}")
    print(f"[OK] Sanity flags: {out['sanity_flags']}")
    print(f"[OK] Missed candidates: {out['missed_candidates']}")
    print(f"[OK] LLM audit JSONL: {out['audit_jsonl']}")
    print(f"[OK] LLM audit CSV:   {out['audit_csv']}")
    print(f"\n[OK] Tudo salvo em: {out['out_dir']}")


if __name__ == "__main__":
    main()
