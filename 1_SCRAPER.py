import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import time

BASE_URL = "https://www.foerderdatenbank.de"

START_URL = (
    "https://www.foerderdatenbank.de/SiteGlobals/FDB/Forms/Suche/Expertensuche_Formular.html?cl2Processes_Foerderart=zuschuss&cl2Processes_Foerdergebiet=_bundesweit+brandenburg&submit=Suchen&cl2Processes_Foerderberechtigte=forschungseinrichtung+hochschule&filterCategories=FundingProgram"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_program_links():
    links = []
    seen = set()
    current_url = START_URL
    page = 1

    while True:
        print(f"[INFO] Página de resultados {page}: {current_url}")

        try:
            resp = requests.get(current_url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.exceptions.ReadTimeout:
            print("[WARN] Timeout na página de resultados. Aguardando e tentando novamente...")
            time.sleep(10)
            continue
        except requests.exceptions.RequestException as e:
            print(f"[ERRO] Falha ao acessar resultados: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        cards = soup.select(
            "div.card.card--horizontal.card--fundingprogram p.card--title a"
        )
        print(f"   -> cards encontrados nesta página: {len(cards)}")

        if not cards:
            break

        for a in cards:
            href = a.get("href")
            if not href:
                continue
            full_url = urljoin(BASE_URL, href)
            if full_url in seen:
                continue
            seen.add(full_url)
            title = a.get_text(strip=True)
            links.append((title, full_url))

        next_link = soup.select_one("div.pagination a.forward.button")
        if not next_link or not next_link.get("href"):
            break

        current_url = urljoin(BASE_URL, next_link["href"])
        page += 1
        time.sleep(2.5)  # mais educado com o servidor

    print(f"[INFO] TOTAL de programas encontrados: {len(links)}")
    return links

def extract_field(dl, label):
    dt = dl.find("dt", string=lambda s: s and label in s)
    if not dt:
        return ""
    dd = dt.find_next_sibling("dd")
    if not dd:
        return ""
    return dd.get_text(" ", strip=True)


def parse_program_detail(title, url):
    print(f"[DETAIL] {title} -> {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    data = {
        "Titel": title,
        "Förderart": "",
        "Förderbereich": "",
        "Fördergebiet": "",
        "Förderberechtigte": "",
        "Fördergeber": "",
        "Link zum Förderprogramm": url,
    }

    # *** CORRIGIDO: grid-modul--two-elements (dois hífens) ***
    dl = soup.select_one(
        "dl.grid-modul--two-elements.document-info-fundingprogram"
    )
    if not dl:
        dl = soup.find("dl")

    if dl:
        data["Förderart"]         = extract_field(dl, "Förderart")
        data["Förderbereich"]     = extract_field(dl, "Förderbereich")
        data["Fördergebiet"]      = extract_field(dl, "Fördergebiet")
        data["Förderberechtigte"] = extract_field(dl, "Förderberechtigte")
        data["Fördergeber"]       = extract_field(dl, "Fördergeber")

    return data


def main():
    programs = get_program_links()

    rows = []
    for i, (title, url) in enumerate(programs, start=1):
        try:
            row = parse_program_detail(title, url)
            rows.append(row)
        except Exception as e:
            print(f"[ERRO] em {url}: {e}")
        time.sleep(0.8)

    csv_filename = "foerderprogramme_286.csv"
    fieldnames = [
        "Titel",
        "Förderart",
        "Förderbereich",
        "Fördergebiet",
        "Förderberechtigte",
        "Fördergeber",
        "Link zum Förderprogramm",
    ]

    with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",            # <<< FIX: separador consistente (Excel DE)
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


    print(f"[OK] CSV criado: {csv_filename}")
    print(f"[OK] Linhas salvas: {len(rows)}")


if __name__ == "__main__":
    main()
