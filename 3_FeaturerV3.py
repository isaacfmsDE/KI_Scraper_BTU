import argparse
import os
import json
import time
import pandas as pd
from openai import OpenAI
import csv  # para configurar separador e quoting
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
print("API KEY FOUND:", bool(os.getenv("OPENAI_API_KEY")))

# ==============================
# CONFIG
# ==============================

MODEL = "gpt-4.1-mini"   # ou "gpt-4.1" se quiser mais robusto
api_key = os.getenv("OPENAI_API_KEY")

# Pasta onde estão os .txt das Richtlinien
TXT_DIR = "richtlinien_txt"
OUTPUT_CSV = "richtlinien_features.csv"


# ==============================
# PROMPT (ajustado com Zuwendungszweck + Förderziel)
# ==============================

PROMPT_TEMPLATE = """
Analysiere den folgenden Richtlinientext zu einem Förderprogramm.

Extrahiere ALLE folgenden Informationen in einem STRIKT validen JSON-Format:

{
  "titel_der_foerderung": "",
  "foerdergeber": "",
  "foerderbereich": "",
  "foerdersumme": null,
  "projekt_max_foerderung": null,
  "foerdernehmer": [],
  "dauer_monate": null,
  "laufzeit_programm_start": "",
  "laufzeit_programm_ende": "",
  "antragsfrist_start": "",
  "antragsfrist_ende": "",
  "antragsdynamik": "",
  "zuwendungszweck": "",
  "foerderziel": "",
  "thematische_schwerpunkte": "",
  "besonderheiten": "",
  "anforderungen": "",
  "thematische_tags": []
}

WICHTIGE REGELN:

- GRUNDREGEL (NICHT RATEN):
  - Wenn ein Datum, eine Fördersumme oder eine Laufzeit NICHT explizit im Text steht,
    darfst du KEINEN Wert raten oder ableiten.
    In diesem Fall MUSS das Feld leer ("") sein – und bei "foerdersumme" MUSS der Wert 0 sein.
  - Verwende KEINE externen Kenntnisse, keine typischen Laufzeiten und keine Annahmen.

- "laufzeit_programm_start" / "laufzeit_programm_ende" (SEHR STRIKT):
  - Trage NUR dann ein Datum ein, wenn der Text ausdrücklich eine Programmlaufzeit/Gültigkeit mit Datum benennt,
    z. B. Formulierungen wie:
    "gilt vom ... bis ...", "in Kraft vom ... bis ...", "Gültigkeit vom ... bis ...",
    "tritt am ... in Kraft", "außer Kraft am ...".
  - Eine Jahreszahl-Spanne wie "2021–2027" oder "2024–2026" zählt NICHT automatisch als Programmlaufzeit,
    wenn nicht klar steht, dass es die Gültigkeit der Richtlinie/Programmlaufzeit ist.
  - Wenn unklar oder nur allgemein beschrieben: beide Felder leer lassen ("").

- "dauer_monate":
  - Wenn die Förderdauer in Jahren oder Wochen angegeben ist, rechne sie in MONATE um.
    Beispiele:
      - 2 Jahre -> 24
      - 18 Monate -> 18
      - 12 Wochen -> ca. 3
  - Wenn der Text nur vage ist (z. B. "in der Regel", "normalerweise", "bis zu"):
    - Gib den Maximalwert an, WENN er explizit genannt ist (z. B. "bis zu 3 Jahre" -> 36).
    - Wenn kein expliziter Zahlenwert genannt ist: lasse "dauer_monate" null.

- "foerdersumme":
  - Normalisiere Geldbeträge zu einer reinen Zahl ohne Währungssymbol.
    Beispiele:
      - "5 Mio. Euro" -> 5000000
      - "100.000 €" -> 100000
  - Wenn Intervalle genannt werden (z. B. 1–10 Mio. €), gib den MAXIMALEN Betrag als "foerdersumme" an.
  - Trage NUR dann einen Betrag ein, wenn im Text ein Betrag explizit genannt ist.
    Wenn KEIN Betrag erkennbar ist: setze "foerdersumme" strikt auf 0.
    
- "projekt_max_foerderung":
    - NUR wenn im Text explizit eine maximale Fördersumme pro Projekt / Vorhaben / Zuwendungsempfänger genannt ist, trage den MAXIMALEN Betrag als Zahl ein (ohne Währung).
    - Wenn nur Programmgesamtbudget genannt ist (Gesamtvolumen) und NICHT pro Projekt: dann 0.
    - Wenn nur Förderquote/Prozentsatz genannt ist und kein Maximalbetrag pro Projekt: dann 0.
    - Wenn nichts eindeutig ist: setze strikt auf 0.

- Datumsfelder ("antragsfrist_start", "antragsfrist_ende"):
  - Wenn möglich IMMER im ISO-Format YYYY-MM-DD.
  - Wenn nur eine Deadline genannt wird, nutze sie als "antragsfrist_ende".
  - Ein "Start" der Antragsfrist darf NUR gesetzt werden, wenn der Text explizit einen Start nennt
    (z. B. "Anträge können ab dem ... gestellt werden").
  - Wenn nichts erkennbar ist, lasse das Feld als leere Zeichenkette "".
  - Wenn im Text nur steht "läuft bis auf Weiteres" oder "gültig ab Veröffentlichung",
    darf KEIN Datum erzeugt werden.
    
- "antragsdynamik" (Antrags-/Call-Mechanik):
    - Beschreibe kurz und strukturiert, WIE Anträge eingereicht werden (z. B. Calls mit Cut-off Dates, laufende Einreichung, mehrstufiges Verfahren).
    - Wenn im Text nur allgemein steht, dass es Calls gibt, aber keine konkreten Termine: erkläre das (z. B. "Termine werden separat veröffentlicht").
    - Wenn es mehrere Runden pro Jahr gibt und das explizit genannt wird: nenne die typische Frequenz.
    - Wenn es relative Fristen gibt (z. B. "x Wochen nach Veröffentlichung"): erwähnen. Falls nicht vorhanden: nicht erfinden.
    - Ausgabe als kurzer strukturierter Text (z. B. 2–5 Zeilen mit '-' oder ';'), kein Roman.

- "foerdernehmer":
  - Gib eine Liste von Kategorien zurück, z. B. ["Unternehmen", "Hochschule", "Forschungseinrichtung"].

- "zuwendungszweck":
  - Beschreibe kurz den unmittelbaren Zweck der Zuwendung (max. 2–3 Sätze).

- "foerderziel":
  - Beschreibe das übergeordnete Ziel / die politische Zielsetzung (max. 2–3 Sätze).

- "thematische_schwerpunkte":
  - Kurze stichpunktartige Zusammenfassung (max. 3–5 Punkte) als EINEN String.

- "besonderheiten":
  - Wichtigste Besonderheiten in max. 2–3 Sätzen.

- "anforderungen":
  - Zentrale Anforderungen in max. 2–3 Sätzen.

- "thematische_tags":
  - Maximal 8 Schlagworte.

- KEIN Fließtext außerhalb des JSON. Antworte NUR mit dem JSON-Objekt.
- Wenn eine Information im Text nicht klar ist: Feld leer lassen ("") oder 0 bei "foerdersumme".
  Erfinde keine Daten.

Hier ist der Richtlinientext:"""



# ==============================
# HELPERS
# ==============================

def to_float_or_none(x):
    
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if not s:
        return None

    # remove símbolos comuns de moeda / milhar
    s = s.replace("€", "").replace("EUR", "")
    s = s.replace(".", "").replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def to_float_or_null(x):
    
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
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


def parse_model_json(raw_content, filename=""):
    """
    Tenta fazer json.loads na resposta do modelo.
    Remove ```json ... ``` se necessário.
    """
    text = raw_content.strip()

    # remove code fences se existirem
    if text.startswith("```"):
        parts = text.split("```")
        candidates = [p for p in parts if "{" in p and "}" in p]
        if candidates:
            text = candidates[0]
            if text.lstrip().lower().startswith("json"):
                first_brace = text.find("{")
                if first_brace != -1:
                    text = text[first_brace:]

    try:
        return json.loads(text)
    except Exception as e:
        print(f"[ERRO] Falha ao parsear JSON para arquivo {filename}: {e}")
        print("Conteúdo recebido do modelo (primeiros 1000 caracteres):")
        print(text[:1000], "...\n")
        return None


# ==============================
# EXTRAÇÃO VIA IA
# ==============================

def build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Defina OPENAI_API_KEY para usar a API da OpenAI.")
    return OpenAI(api_key=api_key)


def extract_features_from_text(client, text, filename=""):
    # monta o prompt sem usar .format
    prompt = PROMPT_TEMPLATE + '\n""" \n' + text + '\n"""'

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content
    data = parse_model_json(raw, filename=filename)
    if data is None:
        return None

    # -------- Normalização dos campos --------
    titel = data.get("titel_der_foerderung", "") or ""
    foerdergeber = data.get("foerdergeber", "") or ""
    foerderbereich = data.get("foerderbereich", "") or ""

    # foerdersumme (único valor numérico)
    foerdersumme = to_float_or_none(data.get("foerdersumme"))

    # se não tiver NENHUM valor -> 0.0 (como você pediu antes)
    if foerdersumme is None:
        foerdersumme = 0.0
        
        
    # projektMaxFoerderung
    
    projekt_max_foerderung = to_float_or_none(data.get("projekt_max_foerderung"))
    if projekt_max_foerderung is None:
        projekt_max_foerderung = 0.0    

    # lista de foerdernehmer
    foerdernehmer_list = ensure_list(data.get("foerdernehmer"))
    foerdernehmer = " | ".join(foerdernehmer_list)

    # duração em meses
    dauer_monate = to_float_or_null(data.get("dauer_monate"))

    laufzeit_programm_start = data.get("laufzeit_programm_start", "") or ""
    laufzeit_programm_ende = data.get("laufzeit_programm_ende", "") or ""
    antragsfrist_start = data.get("antragsfrist_start", "") or ""
    antragsfrist_ende = data.get("antragsfrist_ende", "") or ""
    antragsdynamik = data.get("antragsdynamik", "") or ""

    zuwendungszweck = data.get("zuwendungszweck", "") or ""
    foerderziel = data.get("foerderziel", "") or ""

    thematische_schwerpunkte = data.get("thematische_schwerpunkte", "") or ""
    besonderheiten = data.get("besonderheiten", "") or ""
    anforderungen = data.get("anforderungen", "") or ""

    tags_list = ensure_list(data.get("thematische_tags"))
    thematische_tags = " | ".join(tags_list)

    result = {
        "source_file": filename,
        "titel_der_foerderung": titel,
        "foerdergeber": foerdergeber,
        "foerderbereich": foerderbereich,
        "foerdersumme": foerdersumme,
        "projekt_max_foerderung": projekt_max_foerderung,
        "foerdernehmer": foerdernehmer,
        "dauer_monate": dauer_monate,
        "laufzeit_programm_start": laufzeit_programm_start,
        "laufzeit_programm_ende": laufzeit_programm_ende,
        "antragsfrist_start": antragsfrist_start,
        "antragsfrist_ende": antragsfrist_ende,
        "antragsdynamik": antragsdynamik,
        "zuwendungszweck": zuwendungszweck,
        "foerderziel": foerderziel,
        "thematische_schwerpunkte": thematische_schwerpunkte,
        "besonderheiten": besonderheiten,
        "anforderungen": anforderungen,
        "thematische_tags": thematische_tags,
    }

    return result


# ==============================
# MAIN
# ==============================

def extract_features_from_files(txt_dir, output_csv):
    client = build_client()
    rows = []

    if not os.path.isdir(txt_dir):
        print(f"[ERRO] Pasta '{txt_dir}' não existe. Crie e coloque os .txt lá.")
        return False

    for file in os.listdir(txt_dir):
        if not file.lower().endswith(".txt"):
            continue

        path = os.path.join(txt_dir, file)
        print(f"[PROCESSANDO] {file}")

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        data = extract_features_from_text(client, text, filename=file)

        if data is not None:
            rows.append(data)

        time.sleep(0.8)  # evitar rate limit

    if not rows:
        print("[AVISO] Nenhum dado extraído. Verifique a pasta e os TXT.")
        return False

    df = pd.DataFrame(rows)

    # CSV com separador ';' para não conflitar com vírgulas dentro dos campos
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        sep=";",                 # <<< IMPORTANTE
        quoting=csv.QUOTE_MINIMAL,
    )

    print("\n[OK] Extração concluída!")
    print("[OK] Arquivo gerado:", output_csv)
    return True


def main():
    parser = argparse.ArgumentParser(description="Extrator de features das Richtlinien.")
    parser.add_argument(
        "--txt-dir",
        default=TXT_DIR,
        help="Diretório com arquivos .txt das Richtlinien.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help="Arquivo CSV de saída.",
    )
    args = parser.parse_args()

    extract_features_from_files(args.txt_dir, args.output)


if __name__ == "__main__":
    main()
