# 5_Cleaner.py
# ------------------------------------------------------------
# - Lê:
#   - richtlinien_features.csv
#   - Validation/audit_llm.csv
#   - Validation/sanity_flags.csv
# - Gera:
#   - Results/Programme_Cleaned.csv
#   - Results/cleaning_log.csv
# ------------------------------------------------------------

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_FEATURES = 'richtlinien_features.csv'
DEFAULT_VALIDATION_DIR = 'Validation'
DEFAULT_RESULTS_DIR = 'Results'
DEFAULT_OUT_CSV = 'Programme_Cleaned.csv'
DATE_FIELDS = ['laufzeit_programm_ende', 'antragsfrist_ende']
NUM_FIELDS = ['foerdersumme']
KEEP_COLUMNS = [
    'source_file',
    'titel_der_foerderung',
    'foerdernehmer',
    'laufzeit_programm_ende',
    'antragsfrist_ende',
    'gennante Werte',
    'gennante Friste',
    'antragsdynamik',
    'thematische_schwerpunkte',
    'anforderungen',
    'Link zum Förderprogramm',
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for sep in [',', ';']:
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8-sig', sep=None, engine='python')


def clean_nan_str(v):
    if v is None:
        return ''
    try:
        if isinstance(v, float) and pd.isna(v):
            return ''
    except Exception:
        pass
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def parse_date_to_iso(text: str) -> str:
    t = (text or '').strip()
    if not t:
        return ''
    m = re.search(r'\d{4}-\d{2}-\d{2}', t)
    if m:
        return m.group(0)
    m = re.search(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b', t)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', t)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return ''


def parse_money_to_float(text: str):
    t = (text or '').lower()
    m = re.search(r'(\d+[.,]?\d*)\s*(mio|millionen)', t)
    if m:
        try:
            return float(m.group(1).replace(',', '.')) * 1_000_000
        except Exception:
            return None
    m = re.search(r'(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?)', t)
    if m:
        try:
            return float(m.group(1).replace('.', '').replace(',', '.'))
        except Exception:
            return None
    m = re.search(r'\b(\d+(?:,\d+)?)\b', t)
    if m:
        try:
            return float(m.group(1).replace(',', '.'))
        except Exception:
            return None
    return None


def apply_cleaning(features_df, audit_df, sanity_df):
    df = features_df.copy()
    for c in df.columns:
        df[c] = df[c].apply(clean_nan_str)

    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    if 'gennante Werte' in df.columns:
        df['gennante Werte'] = df['gennante Werte'].apply(
            lambda v: v if clean_nan_str(v) else 'keine gennante Werte'
        )
    if 'gennante Friste' in df.columns:
        df['gennante Friste'] = df['gennante Friste'].apply(
            lambda v: v if clean_nan_str(v) else 'keine gennante friste'
        )

    logs = []

    if not sanity_df.empty and 'flag' in sanity_df.columns:
        bad = sanity_df[sanity_df['flag'] == 'BAD_DATE_FORMAT']
        for _, r in bad.iterrows():
            src = r['source_file']
            m = re.search(r'(laufzeit_programm_ende|antragsfrist_ende)', r.get('detail', ''))
            if not m:
                continue
            field = m.group(1)
            mask = df['source_file'] == src
            if mask.any() and df.loc[mask, field].iloc[0]:
                old = df.loc[mask, field].iloc[0]
                df.loc[mask, field] = ''
                logs.append({'source_file': src, 'field': field, 'action': 'CLEAR_BAD_DATE_FORMAT', 'old': old, 'new': ''})

    if not audit_df.empty:
        for _, r in audit_df.iterrows():
            src = r.get('source_file', '')
            field = r.get('field', '')
            cls = r.get('classification', '')
            found = clean_nan_str(r.get('found_value', '') or r.get('evidence', ''))

            if field not in DATE_FIELDS + NUM_FIELDS + ['gennante Friste']:
                continue

            mask = df['source_file'] == src
            if not mask.any():
                continue
            old = df.loc[mask, field].iloc[0]

            if cls == 'MISMATCH':
                new = '0' if field == 'foerdersumme' else ''
                if field == 'gennante Friste':
                    new = ''
                if old != new:
                    df.loc[mask, field] = new
                    logs.append({'source_file': src, 'field': field, 'action': 'CLEAR_MISMATCH', 'old': old, 'new': new})

            elif cls == 'MISSING_IN_CSV':
                if old:
                    continue
                new = ''
                if field in DATE_FIELDS:
                    new = parse_date_to_iso(found)
                elif field == 'foerdersumme':
                    v = parse_money_to_float(found)
                    new = '' if v is None else str(v)
                elif field == 'gennante Friste' and found:
                    new = found
                if new and old != new:
                    df.loc[mask, field] = new
                    logs.append({'source_file': src, 'field': field, 'action': 'FILL_FROM_TEXT', 'old': old, 'new': new})

    df = df[KEEP_COLUMNS]
    return df, pd.DataFrame(logs)


def run_cleaner(features, validation_dir, results_dir, out_csv):
    base = Path(__file__).resolve().parent
    features_df = read_csv_robust(base / features)
    audit_path = base / validation_dir / 'audit_llm.csv'
    sanity_path = base / validation_dir / 'sanity_flags.csv'
    audit_df = read_csv_robust(audit_path) if audit_path.exists() else pd.DataFrame()
    sanity_df = read_csv_robust(sanity_path) if sanity_path.exists() else pd.DataFrame()

    ensure_dir(base / results_dir)
    cleaned_df, log_df = apply_cleaning(features_df, audit_df, sanity_df)

    cleaned_df.to_csv(base / results_dir / out_csv, sep=';', index=False, encoding='utf-8-sig')
    log_df.to_csv(base / results_dir / 'cleaning_log.csv', index=False, encoding='utf-8-sig')

    print(f'[OK] Cleaned CSV: {results_dir}/{out_csv}')
    print(f'[OK] Cleaning log: {results_dir}/cleaning_log.csv')


def main():
    parser = argparse.ArgumentParser(description='Cleaner – gera Programme_Cleaned.csv')
    parser.add_argument('--features', default=DEFAULT_FEATURES)
    parser.add_argument('--validation-dir', default=DEFAULT_VALIDATION_DIR)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out', '--output', dest='out', default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    run_cleaner(features=args.features, validation_dir=args.validation_dir, results_dir=args.results_dir, out_csv=args.out)


if __name__ == '__main__':
    main()
