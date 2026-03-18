# 2_Classifier_V3.py  (corrigido: CSV sep=";", prompt OK, sem chave hardcoded, sem docstring quebrada)
# - Lê corretamente o CSV do scraper (sep=";")
# - Mantém fallback para CSV “colapsado” em 1 coluna (caso alguém salve errado)
# - Salva saída também com sep=";" (Excel DE)
# - NÃO guarda API key no código (usa OPENAI_API_KEY via ambiente/.env)

import argparse
import os
import time
import json
import csv
import pandas as pd
from openai import OpenAI

try:
    # opcional: se você usa .env no projeto
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# ==============================
# CONFIG
# ==============================
MODEL = "gpt-4.1-mini"


# ==============================
# INSTRUÇÕES (PROMPT)
# ==============================
CLASSIFIER_INSTRUCTIONS = """
Du bist ein Fördermittel-Experte für den Lehrstuhl Produktionswirtschaft der BTU Cottbus–Senftenberg.

AUFGABE
Du bekommst den Text einer Förderrichtlinie oder Programmausschreibung (oder eine Kurzfassung) und sollst entscheiden,
ob das Programm für unseren Lehrstuhl in der PROSPEKTION geeignet ist oder nicht.

WICHTIGE ENTSCHEIDUNGSREGEL (PROSPEKTION, NICHT FINAL)
Dieses Tool dient der frühen Programmsichtung. Daher gilt:

- Wenn die thematische Passung anhand des TITELS oder des PROGRAMMNAMENS klar auf industrielle Wertschöpfung,
  Produktion, Fertigung, Logistik/Supply Chain, Industrie 4.0, Digitalisierung, Datenräume, KI/ML, Simulation
  oder verwandte industrielle Innovations-/Transformationsfelder hinweist, dann ist das Programm als
  GEEIGNET (verdict="JA") zu bewerten – auch wenn Details zur Förderquote oder Rolle der Hochschule im Text fehlen.

- Ein Programm darf NUR dann mit "NEIN" bewertet werden, wenn:
  (a) Titel und Kontext klar fachfremd sind (z. B. Agrar/Bioökonomie ohne Produktionsbezug, Medizin/klinisch,
      Schul-/Jugendprogramme, reine Hochschulpolitik/Governance, reine kommunale Sozialprogramme ohne Industriebezug),
  ODER
  (b) der Text explizit sagt, dass Hochschulen/Forschungseinrichtungen NICHT antragsberechtigt sind,
      oder keine eigene Förderung erhalten.

MERKSATZ:
Im Zweifel "JA", solange der Titel thematisch passt und keine explizite Ausschlussregel vorliegt.

PROFIL DES LEHRSTUHLS
- Deutsche Universität / Hochschule, Lehrstuhl Produktionswirtschaft.
- Themen (stark vereinfacht):
  - Produktionsmanagement, Produktions- und Logistiksysteme, Wertschöpfungsnetzwerke.
  - Industrie 4.0, digitale Transformation, digitale Fabrik, cyber-physische Produktionssysteme.
  - Cloud Manufacturing / Manufacturing-as-a-Service, Plattformökonomie, Datenräume, Interoperabilität.
  - Digitale Zwillinge, Simulation, KI/ML in Produktion und Logistik, Entscheidungsunterstützung.
  - Geschäftsmodelle, Technologie- und Innovationsmanagement im industriellen Kontext.
- Typische Rolle: Forschungs- und Entwicklungsprojekte (FuE) mit Industriepartnern; keine agrar-, bio-, medizinische
  oder rein hochschulpolitische Forschung.

KRITERIEN FÜR „GEEIGNET“ (im Prospektions-Sinn)
Ein Programm ist GEEIGNET (verdict="JA"), wenn:
- Thematische Passung gegeben ist (mindestens anhand Titel/Programmname erkennbar) UND
- KEIN expliziter Ausschluss der Hochschule/Forschungseinrichtung als Zuwendungsempfänger im Text steht.

Hinweis:
- Es ist in Ordnung, wenn Industrie-/KMU-Partner oder ausländische Partner verpflichtend sind.
- Fristen, genaue Budgethöhe und Laufzeit ignorierst du.

ANTWORTFORMAT
Gib deine Antwort ausschließlich als JSON-Objekt in einer Zeile zurück:
{"verdict":"JA","begruendung":"<maximal 1–2 sehr kurze Sätze auf Deutsch, warum JA oder NEIN>"}

Regeln:
- "verdict" ist immer "JA" oder "NEIN".
- Bei "JA": Begründe kurz mit thematischer Passung (ggf. anhand Titel).
- Bei "NEIN": Begründe kurz mit klarer Nicht-Passung oder explizitem Ausschluss der Hochschule.
- Keine weiteren Felder, kein zusätzlicher Text außerhalb des JSON.
""".strip()


# ==============================
# IO HELPERS
# ==============================
def load_csv_semicolon_or_fallback(path: str) -> pd.DataFrame:
    """
    Lê CSV do scraper (sep=';').
    Se estiver “colapsado” em 1 coluna (ex.: Excel salvou errado),
    faz split manual por ';'.
    """
    # tentativa normal (o esperado)
    df = pd.read_csv(path, encoding="utf-8-sig", sep=";")
    if df.shape[1] > 1:
        return df

    # fallback: 1 coluna com tudo concatenado
    df_raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    col = df_raw.iloc[:, 0].astype(str)

    header_str = col.iloc[0]
    headers = [h.strip() for h in header_str.split(";") if h.strip()]

    data_rows = col.iloc[1:]
    data = data_rows.str.split(";", expand=True)
    data.columns = headers
    return data.reset_index(drop=True)


def load_xlsx_smart(path: str) -> pd.DataFrame:
    """
    Lê um XLSX no formato “1 coluna com tudo separado por ;”.
    (mantido para compatibilidade com seu fluxo antigo)
    """
    df_raw = pd.read_excel(path, header=None)
    col = df_raw.iloc[:, 0].astype(str)

    header_str = col.iloc[0]
    headers = [h.strip() for h in header_str.split(";") if h.strip()]

    data_rows = col.iloc[1:]
    df = data_rows.str.split(";", expand=True)
    df.columns = headers
    return df.reset_index(drop=True)


def load_program_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        print(f"[INFO] Lendo Excel: {path}")
        return load_xlsx_smart(path)

    if path.lower().endswith(".csv"):
        print(f"[INFO] Lendo CSV: {path}")
        return load_csv_semicolon_or_fallback(path)

    raise ValueError("Arquivo não suportado. Use .csv ou .xlsx.")


# ==============================
# OPENAI
# ==============================
def build_client() -> OpenAI:
    if load_dotenv is not None:
        # opcional: carrega .env na pasta atual
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY não encontrada. Defina a variável de ambiente ou use um arquivo .env."
        )
    return OpenAI(api_key=api_key)


def parse_json_safely(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        candidates = [p for p in parts if "{" in p and "}" in p]
        if candidates:
            s = candidates[0].strip()
            if s.lower().startswith("json"):
                s = s[s.find("{") :]
    return json.loads(s)


def classify_program(client: OpenAI, program_data: dict) -> tuple[str, str]:
    program_text = f"""Titel: {program_data.get('Titel','')}
Förderart: {program_data.get('Förderart','')}
Förderbereich: {program_data.get('Förderbereich','')}
Fördergebiet: {program_data.get('Fördergebiet','')}
Förderberechtigte: {program_data.get('Förderberechtigte','')}
Fördergeber: {program_data.get('Fördergeber','')}
"""

    prompt = (
        CLASSIFIER_INSTRUCTIONS
        + "\n\nPROGRAMMTEXT (Kurzfassung des Förderprogramms):\n"
        + program_text
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = (resp.choices[0].message.content or "").strip()

    try:
        data = parse_json_safely(raw)
        verdict = str(data.get("verdict", "")).strip().upper()
        begruendung = str(data.get("begruendung", "")).strip()
    except Exception:
        verdict = "ERRO"
        begruendung = "Fehler beim Parsen der Modellantwort."

    if verdict not in ("JA", "NEIN", "ERRO"):
        verdict = "ERRO"

    return verdict, begruendung


# ==============================
# MAIN
# ==============================
def run_classification(input_path: str, output_path: str, sleep_s: float = 0.7) -> None:
    client = build_client()
    df = load_program_data(input_path)

    if "Titel" not in df.columns:
        raise RuntimeError(f"A coluna 'Titel' não foi encontrada! Colunas: {list(df.columns)}")

    labels, begruendungen = [], []

    for _, row in df.iterrows():
        title = str(row.get("Titel", "")).strip()
        print(f"[CLASSIFICANDO] {title}")

        try:
            verdict, begruendung = classify_program(client, row.to_dict())
        except Exception as e:
            print(f"[ERRO] {title} -> {e}")
            verdict, begruendung = "ERRO", "Fehler während der Klassifikation."

        labels.append(verdict)
        begruendungen.append(begruendung)
        time.sleep(sleep_s)

    df["Label"] = labels
    df["Begruendung"] = begruendungen

    # IMPORTANT: salva com sep=";" para consistência com o pipeline e Excel DE
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        quoting=csv.QUOTE_MINIMAL,
    )

    print("\n[OK] Classificação concluída!")
    print(f"[OK] Arquivo gerado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classificador de programas.")
    parser.add_argument("--input", default="foerderprogramme_286.csv", help="Arquivo de entrada (.csv ou .xlsx).")
    parser.add_argument("--output", default="foerderprogramme_Labeled.csv", help="Arquivo de saída (.csv).")
    parser.add_argument("--sleep", type=float, default=0.7, help="Pausa entre requisições (segundos).")
    args = parser.parse_args()

    run_classification(args.input, args.output, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
