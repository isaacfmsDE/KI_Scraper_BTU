# pipeline.py
import argparse
import importlib.util
import os
import time
import csv
import subprocess
import sys

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from richtlinie_extractor import extract_richtlinien_from_csv


def load_module(module_name, filename):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scraper_module():
    return load_module("scraper_module", "1_SCRAPER.py")


def load_classifier_module():
    return load_module("classifier_module", "2_Classifier_V3.py")


def load_featurer_module():
    return load_module("featurer_module", "3_FeaturerV3.py")


def run_scraper(output_csv):
    scraper = load_scraper_module()
    programs = scraper.get_program_links()

    rows = []
    for title, url in programs:
        try:
            row = scraper.parse_program_detail(title, url)
            rows.append(row)
        except Exception as exc:
            print(f"[ERRO] em {url}: {exc}")
        time.sleep(0.8)

    fieldnames = [
        "Titel",
        "Förderart",
        "Förderbereich",
        "Fördergebiet",
        "Förderberechtigte",
        "Fördergeber",
        "Link zum Förderprogramm",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV criado: {output_csv}")
    print(f"[OK] Linhas salvas: {len(rows)}")


def run_validator(
    scraper_csv,
    labeled_csv,
    features_csv,
    txt_dir,
    validation_dir="Validation"
):
    """
    Executa 4_Validator.py como subprocess.
    Por padrão, ele deve rodar o fluxo automático que você implementou
    (suspicious + jsonl->csv etc). Se o seu 4_Validator ainda depender
    de flags, ajuste aqui.
    """
    cmd = [
        sys.executable, "4_Validator.py",
        "--scraper", scraper_csv,
        "--labeled", labeled_csv,
        "--features", features_csv,
        "--txt-dir", txt_dir,
        "--out-dir", validation_dir,
    ]
    print("[INFO] Rodando validator:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_cleaner(
    features_csv,
    validation_dir="Validation",
    results_dir="Results",
    output_csv="Programme_Cleaned.csv"
):
    """
    Executa 5_Cleaner.py como subprocess.
    """
    cmd = [
        sys.executable, "5_Cleaner.py",
        "--features", features_csv,
        "--validation-dir", validation_dir,
        "--results-dir", results_dir,
        "--output", output_csv,
    ]
    print("[INFO] Rodando cleaner:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo do scraper + validator + cleaner.")
    parser.add_argument("--scraper-output", default="foerderprogramme_286.csv")
    parser.add_argument("--labeled-output", default="foerderprogramme_Labeled.csv")
    parser.add_argument("--txt-dir", default="richtlinien_txt")
    parser.add_argument("--features-output", default="richtlinien_features.csv")

    parser.add_argument("--validation-dir", default="Validation")
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--cleaned-output", default="Programme_Cleaned.csv")

    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-classify", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-featurize", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")

    args = parser.parse_args()

    if not args.skip_scrape:
        run_scraper(args.scraper_output)

    if not args.skip_classify:
        classifier = load_classifier_module()
        classifier.run_classification(args.scraper_output, args.labeled_output)

    if not args.skip_download:
        extract_richtlinien_from_csv(args.labeled_output, args.txt_dir)

    if not args.skip_featurize:
        featurer = load_featurer_module()
        featurer.extract_features_from_files(args.txt_dir, args.features_output, args.labeled_output)

    if not args.skip_validate:
        run_validator(
            scraper_csv=args.scraper_output,
            labeled_csv=args.labeled_output,
            features_csv=args.features_output,
            txt_dir=args.txt_dir,
            validation_dir=args.validation_dir,
        )

    if not args.skip_clean:
        run_cleaner(
            features_csv=args.features_output,
            validation_dir=args.validation_dir,
            results_dir=args.results_dir,
            output_csv=args.cleaned_output,
        )

    print("\n[OK] Pipeline finalizada.")
    print(f"[OK] Validation dir: {Path(args.validation_dir).resolve()}")
    print(f"[OK] Results dir: {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
