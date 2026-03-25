import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).with_name('.env'))
print('API KEY FOUND:', bool(os.getenv('OPENAI_API_KEY')))

MODEL = 'gpt-4.1-mini'
TXT_DIR = 'richtlinien_txt'
OUTPUT_CSV = 'richtlinien_features.csv'
DEFAULT_SOURCE_CSV = 'foerderprogramme_Labeled.csv'
NO_DEADLINE_TEXT = 'keine gennante friste'
NO_VALUES_TEXT = 'keine gennante Werte'
OUTPUT_COLUMNS = [
    'source_file',
    'titel_der_foerderung',
    'foerdersumme',
    'projekt_max_foerderung',
    'foerdernehmer',
    'laufzeit_programm_ende',
    'antragsfrist_ende',
    'antragsdynamik',
    'gennante Werte',
    'gennante Friste',
    'thematische_schwerpunkte',
    'anforderungen',
    'Link zum Förderprogramm',
]

PROMPT_TEMPLATE = """
Analysiere den folgenden Richtlinientext zu einem Förderprogramm.

Extrahiere ALLE folgenden Informationen in einem STRIKT validen JSON-Format:

{
  "titel_der_foerderung": "",
  "foerdersumme": null,
  "projekt_max_foerderung": null,
  "foerdernehmer": [],
  "laufzeit_programm_ende": "",
  "antragsfrist_ende": "",
  "antragsdynamik": "",
  "gennante_werte": [],
  "thematische_schwerpunkte": "",
  "anforderungen": "",
  "gennante_fristen": []
}

WICHTIGE REGELN:

- GRUNDREGEL (NICHT RATEN):
  - Wenn ein Datum oder eine Fördersumme NICHT explizit im Text steht,
    darfst du KEINEN Wert raten oder ableiten.
  - Verwende KEINE externen Kenntnisse, keine typischen Laufzeiten und keine Annahmen.

- "foerdersumme":
  - Normalisiere Geldbeträge zu einer reinen Zahl ohne Währungssymbol.
    Beispiele:
      - "5 Mio. Euro" -> 5000000
      - "100.000 €" -> 100000
  - Wenn Intervalle genannt werden (z. B. 1–10 Mio. €), gib den MAXIMALEN Betrag als "foerdersumme" an.
  - Trage NUR dann einen Betrag ein, wenn im Text ein Betrag explizit genannt ist.
    Wenn KEIN Betrag erkennbar ist: setze "foerdersumme" strikt auf 0.

- "projekt_max_foerderung":
  - NUR wenn im Text explizit eine maximale Fördersumme pro Projekt / Vorhaben / Zuwendungsempfänger genannt ist,
    trage den MAXIMALEN Betrag als Zahl ein (ohne Währung).
  - Wenn nur Programmgesamtbudget genannt ist und NICHT pro Projekt: dann 0.
  - Wenn nichts eindeutig ist: setze strikt auf 0.

- Datumsfelder:
  - Verwende wenn möglich das ISO-Format YYYY-MM-DD.
  - "laufzeit_programm_ende" darf nur gesetzt werden, wenn das Ende der Programmlaufzeit / Gültigkeit explizit genannt ist.
  - "antragsfrist_ende" darf gesetzt werden, wenn eine explizite Deadline / Einreichungsfrist genannt wird.
  - Wenn nichts erkennbar ist, lasse das Feld leer als "".

- "antragsdynamik":
  - Beschreibe kurz und strukturiert, WIE Anträge eingereicht werden.
  - Wenn keine klare Information vorliegt: leer lassen.

- "foerdernehmer":
  - Gib eine Liste von Kategorien zurück, z. B. ["Unternehmen", "Hochschule", "Forschungseinrichtung"].

- "thematische_schwerpunkte":
  - Kurze stichpunktartige Zusammenfassung (max. 3–5 Punkte) als EINEN String.

- "anforderungen":
  - Zentrale Anforderungen in max. 2–3 Sätzen.

- "gennante_fristen":
  - Sammle ALLE explizit im Text genannten Datumsangaben mit kurzer Bedeutung.
  - Ausgabe als JSON-Liste von Strings.
  - Jeder Eintrag muss dem Muster folgen: "<Datum> - <kurze Bedeutung>".
  - Beispiele:
    - "14/03/2027 - Frist zur Abgabe der Antragsunterlage"
    - "21.05.2027 - Stichtag für die zweite Auswahlrunde"
  - Wenn mehrere Daten genannt werden, gib alle zurück.
  - Wenn es absolut keine explizit genannten Daten im Text gibt, gib eine leere Liste [] zurück.
  - Erfinde keine Bedeutungen; nutze nur kurze, textnahe Beschreibungen.

- "gennante_werte":
  - Sammle ALLE explizit im Text genannten EURO-Beträge mit kurzer Bedeutung.
  - Ausgabe als JSON-Liste von Strings.
  - Jeder Eintrag muss dem Muster folgen: "<Wert> - <kurze Bedeutung>".
  - Beispiele:
    - "5 Mio. Euro - Programmbudget"
    - "200.000 € - Maximale Förderung pro Projekt"
  - Wenn mehrere Werte genannt werden, gib alle zurück.
  - Wenn es keine explizit genannten EURO-Werte gibt, gib eine leere Liste [] zurück.
  - Erfinde keine Bedeutungen; nutze nur kurze, textnahe Beschreibungen.

- KEIN Fließtext außerhalb des JSON. Antworte NUR mit dem JSON-Objekt.
"""


def to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if not s:
        return None

    s = s.replace('€', '').replace('EUR', '')
    s = s.replace('.', '').replace(' ', '')
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def normalize_deadline_entries(entries):
    cleaned = []
    for entry in ensure_list(entries):
        line = re.sub(r'\s+', ' ', entry).strip()
        if line:
            cleaned.append(line)
    return cleaned


def normalize_value_entries(entries):
    cleaned = []
    for entry in ensure_list(entries):
        line = re.sub(r'\s+', ' ', entry).strip()
        if line:
            cleaned.append(line)
    return cleaned


def parse_model_json(raw_content, filename=''):
    text = raw_content.strip()
    if text.startswith('```'):
        parts = text.split('```')
        candidates = [p for p in parts if '{' in p and '}' in p]
        if candidates:
            text = candidates[0]
            if text.lstrip().lower().startswith('json'):
                first_brace = text.find('{')
                if first_brace != -1:
                    text = text[first_brace:]

    try:
        return json.loads(text)
    except Exception as e:
        print(f'[ERRO] Falha ao parsear JSON para arquivo {filename}: {e}')
        print('Conteúdo recebido do modelo (primeiros 1000 caracteres):')
        print(text[:1000], '...\n')
        return None


def build_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError('Defina OPENAI_API_KEY para usar a API da OpenAI.')
    return OpenAI(api_key=api_key)


def build_slug_to_link_map(source_csv):
    if not source_csv or not os.path.exists(source_csv):
        return {}

    try:
        df = pd.read_csv(source_csv, encoding='utf-8-sig', sep=';')
    except Exception:
        df = pd.read_csv(source_csv, encoding='utf-8-sig')

    if 'Titel' not in df.columns or 'Link zum Förderprogramm' not in df.columns:
        return {}

    slug_map = {}
    for _, row in df.iterrows():
        titel = str(row.get('Titel', '') or '').strip()
        link = str(row.get('Link zum Förderprogramm', '') or '').strip()
        if not titel or not link:
            continue
        slug = slugify_filename(titel)
        slug_map.setdefault(slug, link)
    return slug_map


def slugify_filename(text):
    value = str(text or '').lower().strip()
    value = re.sub(r"[^\w\s-]", '', value)
    value = re.sub(r'[\s_-]+', '-', value)
    return value[:100] or 'programm'


def link_for_source_file(filename, slug_map):
    stem = Path(filename).stem
    return slug_map.get(stem, '')


def extract_features_from_text(client, text, filename=''):
    prompt = PROMPT_TEMPLATE + '\n"""\n' + text + '\n"""'

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content
    data = parse_model_json(raw, filename=filename)
    if data is None:
        return None

    titel = data.get('titel_der_foerderung', '') or ''
    foerdersumme = to_float_or_none(data.get('foerdersumme'))
    if foerdersumme is None:
        foerdersumme = 0.0

    projekt_max_foerderung = to_float_or_none(data.get('projekt_max_foerderung'))
    if projekt_max_foerderung is None:
        projekt_max_foerderung = 0.0

    foerdernehmer = ' | '.join(ensure_list(data.get('foerdernehmer')))
    laufzeit_programm_ende = data.get('laufzeit_programm_ende', '') or ''
    antragsfrist_ende = data.get('antragsfrist_ende', '') or ''
    antragsdynamik = data.get('antragsdynamik', '') or ''
    thematische_schwerpunkte = data.get('thematische_schwerpunkte', '') or ''
    anforderungen = data.get('anforderungen', '') or ''

    werte = normalize_value_entries(data.get('gennante_werte'))
    gennante_werte = '\n'.join(werte) if werte else NO_VALUES_TEXT

    fristen = normalize_deadline_entries(data.get('gennante_fristen'))
    gennante_fristen = '\n'.join(fristen) if fristen else NO_DEADLINE_TEXT

    return {
        'source_file': filename,
        'titel_der_foerderung': titel,
        'foerdersumme': foerdersumme,
        'projekt_max_foerderung': projekt_max_foerderung,
        'foerdernehmer': foerdernehmer,
        'laufzeit_programm_ende': laufzeit_programm_ende,
        'antragsfrist_ende': antragsfrist_ende,
        'antragsdynamik': antragsdynamik,
        'gennante Werte': gennante_werte,
        'gennante Friste': gennante_fristen,
        'thematische_schwerpunkte': thematische_schwerpunkte,
        'anforderungen': anforderungen,
    }


def extract_features_from_files(txt_dir, output_csv, source_csv=DEFAULT_SOURCE_CSV):
    client = build_client()
    rows = []

    if not os.path.isdir(txt_dir):
        print(f"[ERRO] Pasta '{txt_dir}' não existe. Crie e coloque os .txt lá.")
        return False

    slug_map = build_slug_to_link_map(source_csv)

    for file in sorted(os.listdir(txt_dir)):
        if not file.lower().endswith('.txt'):
            continue

        path = os.path.join(txt_dir, file)
        print(f'[PROCESSANDO] {file}')

        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()

        data = extract_features_from_text(client, text, filename=file)

        if data is not None:
            data['Link zum Förderprogramm'] = link_for_source_file(file, slug_map)
            rows.append(data)

        time.sleep(0.8)

    if not rows:
        print('[AVISO] Nenhum dado extraído. Verifique a pasta e os TXT.')
        return False

    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    df = df[OUTPUT_COLUMNS]

    df.to_csv(
        output_csv,
        index=False,
        encoding='utf-8-sig',
        sep=';',
        quoting=csv.QUOTE_MINIMAL,
    )

    print('\n[OK] Extração concluída!')
    print('[OK] Arquivo gerado:', output_csv)
    return True


def main():
    parser = argparse.ArgumentParser(description='Extrator de features das Richtlinien.')
    parser.add_argument('--txt-dir', default=TXT_DIR, help='Diretório com arquivos .txt das Richtlinien.')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Arquivo CSV de saída.')
    parser.add_argument('--source-csv', default=DEFAULT_SOURCE_CSV, help='CSV com Titel e Link zum Förderprogramm.')
    args = parser.parse_args()

    extract_features_from_files(args.txt_dir, args.output, args.source_csv)


if __name__ == '__main__':
    main()
