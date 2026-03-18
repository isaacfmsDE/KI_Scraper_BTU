# 5_Cleaner.py  (Scraper_V4)
# ------------------------------------------------------------
# - Lê:
#   - richtlinien_features.csv
#   - Validation/audit_llm.csv
#   - Validation/sanity_flags.csv
#
# - Gera:
#   - Results/Programme_Cleaned.csv
#   - Results/cleaning_log.csv
#
# Política (CONSERVADORA, SEM INVENTAR DADOS):
# - MISMATCH            -> limpa campo ("" ou 0 para foerdersumme)
# - BAD_DATE_FORMAT     -> limpa campo de data
# - MISSING_IN_CSV      -> só preenche se parse for determinístico
# - MISSING_IN_TEXT     -> mantém vazio (comportamento correto)
# ------------------------------------------------------------

import re
import argparse
from pathlib import Path
import pandas as pd


# ==============================
# DEFAULTS
# ==============================
DEFAULT_FEATURES = "richtlinien_features.csv"
DEFAULT_VALIDATION_DIR = "Validation"
DEFAULT_RESULTS_DIR = "Results"
DEFAULT_OUT_CSV = "Programme_Cleaned.csv"


DATE_FIELDS = [
    "laufzeit_programm_start",
    "laufzeit_programm_ende",
    "antragsfrist_start",
    "antragsfrist_ende",
]

NUM_FIELDS = [
    "foerdersumme",
    "dauer_monate",
]


GERMAN_MONTHS = {
    "januar": "01", "jan": "01",
    "februar": "02", "feb": "02",
    "märz": "03", "maerz": "03", "mrz": "03",
    "april": "04", "apr": "04",
    "mai": "05",
    "juni": "06", "jun": "06",
    "juli": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09",
    "oktober": "10", "okt": "10",
    "november": "11", "nov": "11",
    "dezember": "12", "dez": "12",
}


# ==============================
# UTILS
# ==============================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


def clean_nan_str(v):
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def is_iso_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""))


# ==============================
# PARSERS (DETERMINÍSTICOS)
# ==============================
def parse_date_to_iso(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if m:
        return m.group(0)

    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", t)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    m = re.search(r"\b(\d{1,2})\.\s*([A-Za-zÄÖÜäöüß]+)\s*(\d{4})\b", t)
    if m:
        day = int(m.group(1))
        mon = m.group(2).lower()
        mon = mon.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        year = int(m.group(3))
        mm = GERMAN_MONTHS.get(mon)
        if mm:
            return f"{year:04d}-{int(mm):02d}-{day:02d}"

    return ""


def parse_money_to_float(text: str):
    t = (text or "").lower()

    m = re.search(r"(\d+[.,]?\d*)\s*(mio|millionen)", t)
    if m:
        try:
            return float(m.group(1).replace(",", ".")) * 1_000_000
        except Exception:
            return None

    m = re.search(r"(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?)", t)
    if m:
        try:
            return float(m.group(1).replace(".", "").replace(",", "."))
        except Exception:
            return None

    m = re.search(r"\b(\d+(?:,\d+)?)\b", t)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    return None


def parse_duration_months(text):
    # Se já for número, retorna direto
    if isinstance(text, (int, float)):
        try:
            return int(round(float(text)))
        except Exception:
            return None

    # Se não for string, aborta
    if not isinstance(text, str):
        return None

    t = text.lower()

    # padrões comuns
    m = re.search(r"(\d+)\s*(monat|monate|months?)", t)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*(jahr|jahre|years?)", t)
    if m:
        return int(m.group(1)) * 12

    return None



# ==============================
# CORE CLEANING
# ==============================
def apply_cleaning(features_df, audit_df, sanity_df):
    df = features_df.copy()

    for c in df.columns:
        df[c] = df[c].apply(clean_nan_str)

    logs = []

    # --- SANITY FLAGS ---
    if not sanity_df.empty:
        bad = sanity_df[sanity_df["flag"] == "BAD_DATE_FORMAT"]
        for _, r in bad.iterrows():
            src = r["source_file"]
            m = re.search(r"(laufzeit_programm_start|laufzeit_programm_ende|antragsfrist_start|antragsfrist_ende)", r["detail"])
            if not m:
                continue
            field = m.group(1)
            mask = df["source_file"] == src
            if mask.any() and df.loc[mask, field].iloc[0]:
                old = df.loc[mask, field].iloc[0]
                df.loc[mask, field] = ""
                logs.append({"source_file": src, "field": field, "action": "CLEAR_BAD_DATE_FORMAT", "old": old, "new": ""})

    # --- AUDIT RULES ---
    if not audit_df.empty:
        for _, r in audit_df.iterrows():
            src = r["source_file"]
            field = r["field"]
            cls = r["classification"]
            found = r.get("found_value", "") or r.get("evidence", "")

            if field not in DATE_FIELDS + NUM_FIELDS:
                continue

            mask = df["source_file"] == src
            if not mask.any():
                continue

            old = df.loc[mask, field].iloc[0]

            if cls == "MISMATCH":
                new = "0" if field == "foerdersumme" else ""
                if old != new:
                    df.loc[mask, field] = new
                    logs.append({"source_file": src, "field": field, "action": "CLEAR_MISMATCH", "old": old, "new": new})

      
            elif cls == "MISSING_IN_CSV":
                # Só preenche se o CSV estiver realmente vazio
                if old:
                    continue
            
                found_value = clean_nan_str(r.get("found_value", ""))
                evidence = clean_nan_str(r.get("evidence", ""))
                found = found_value if found_value else evidence
            
                new = ""
                if field in DATE_FIELDS:
                    new = parse_date_to_iso(found)
                elif field == "foerdersumme":
                    v = parse_money_to_float(found)
                    new = "" if v is None else str(v)
                elif field == "dauer_monate":
                    v = parse_duration_months(found)
                    new = "" if v is None else str(v)
            
                if new and old != new:
                    df.loc[mask, field] = new
                    logs.append({"source_file": src, "field": field, "action": "FILL_FROM_TEXT", "old": old, "new": new})


    return df, pd.DataFrame(logs)


# ==============================
# RUNNER
# ==============================
def run_cleaner(features, validation_dir, results_dir, out_csv):
    base = Path(__file__).resolve().parent

    features_df = read_csv_robust(base / features)
    audit_df = read_csv_robust(base / validation_dir / "audit_llm.csv") if (base / validation_dir / "audit_llm.csv").exists() else pd.DataFrame()
    sanity_df = read_csv_robust(base / validation_dir / "sanity_flags.csv") if (base / validation_dir / "sanity_flags.csv").exists() else pd.DataFrame()

    ensure_dir(base / results_dir)

    cleaned_df, log_df = apply_cleaning(features_df, audit_df, sanity_df)

    cleaned_df.to_csv(base / results_dir / out_csv, sep=";", index=False, encoding="utf-8-sig")
    log_df.to_csv(base / results_dir / "cleaning_log.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Cleaned CSV: {results_dir}/{out_csv}")
    print(f"[OK] Cleaning log: {results_dir}/cleaning_log.csv")


def main():
    parser = argparse.ArgumentParser(description="Cleaner – gera Programme_Cleaned.csv")

    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--validation-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)

    # ACEITA --out E --output
    parser.add_argument("--out", "--output", dest="out", default=DEFAULT_OUT_CSV)

    args = parser.parse_args()

    run_cleaner(
        features=args.features,
        validation_dir=args.validation_dir,
        results_dir=args.results_dir,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
