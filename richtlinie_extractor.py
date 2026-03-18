import os
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

KEYWORDS = [
    "richtlinie",
    "förderrichtlinie",
    "bekanntmachung",
    "rechtsgrundlage",
]


def slugify(text):
    value = text.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value)
    return value[:100] or "programm"

        

def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def has_keyword(text):
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in KEYWORDS)


def find_section_container(heading):
    container = heading.find_parent(
        class_=lambda c: c and ("rich-text" in c or "content--tab-text" in c)
    )
    return container or heading.parent


def find_rechtsgrundlage_sections(soup):
    sections = []
    for heading in soup.find_all(["h2", "h3", "h4", "h5"]):
        if has_keyword(heading.get_text(" ", strip=True)):
            container = find_section_container(heading)
            if container and container not in sections:
                sections.append(container)
    return sections


def extract_text_from_section(section):
    text = section.get_text("\n", strip=True)
    return text


def find_links(section):
    links = []
    for link in section.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(" ", strip=True)
        links.append((href, text))
    return links


def filter_candidate_links(links):
    candidates = []
    for href, text in links:
        combined = f"{href} {text}".lower()
        if ".pdf" in href.lower() or any(keyword in combined for keyword in KEYWORDS):
            candidates.append((href, text))
    return candidates


def fetch_url(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def download_file_from_response(resp, dest_path):
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def pdf_to_text(path):
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None
    if pdfplumber:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_html_text_from_response(resp):
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.select_one("main") or soup.body
    return main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)


def save_text(content, dest_path):
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_richtlinie_for_program(title, url, output_dir):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sections = find_rechtsgrundlage_sections(soup)
    links = []
    for section in sections:
        links.extend(filter_candidate_links(find_links(section)))

    if not links:
        page_links = filter_candidate_links(
            [(a["href"], a.get_text(" ", strip=True)) for a in soup.find_all("a", href=True)]
        )
        links.extend(page_links)

    filename_base = slugify(title)
    os.makedirs(output_dir, exist_ok=True)

    for href, _ in links:
        full_url = urljoin(url, href)
        resp = fetch_url(full_url)
        content_type = resp.headers.get("Content-Type", "").lower()
        is_pdf = "pdf" in content_type or ".pdf" in full_url.lower()

        if is_pdf:
            pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
            download_file_from_response(resp, pdf_path)
            text = pdf_to_text(pdf_path)
            if text.strip():
                txt_path = os.path.join(output_dir, f"{filename_base}.txt")
                save_text(text, txt_path)
                return txt_path
        else:
            text = fetch_html_text_from_response(resp)
            if text.strip():
                txt_path = os.path.join(output_dir, f"{filename_base}.txt")
                save_text(text, txt_path)
                return txt_path

    for section in sections:
        text = extract_text_from_section(section)
        if text.strip():
            txt_path = os.path.join(output_dir, f"{filename_base}.txt")
            save_text(text, txt_path)
            return txt_path

    return None


def extract_richtlinien_from_csv(labeled_csv, output_dir, label_column="Label"):
    df = pd.read_csv(labeled_csv, encoding="utf-8-sig", sep=";")
    if "Titel" not in df.columns or "Link zum Förderprogramm" not in df.columns:
        raise Exception("CSV precisa conter 'Titel' e 'Link zum Förderprogramm'.")

    selected = df
    if label_column in df.columns:
        selected = df[df[label_column].astype(str).str.upper() == "JA"]


    txt_paths = []
    for _, row in selected.iterrows():
        title = row["Titel"]
        url = row["Link zum Förderprogramm"]
        print(f"[RICHTLINIE] {title} -> {url}")
        try:
            txt_path = extract_richtlinie_for_program(title, url, output_dir)
            if txt_path:
                txt_paths.append(txt_path)
            else:
                print(f"[AVISO] Nenhuma Richtlinie encontrada para {title}")
        except Exception as exc:
            print(f"[ERRO] {title}: {exc}")
        time.sleep(1.0)

    print(f"[DEBUG] Total rows CSV: {len(df)}")
    print(f"[DEBUG] Rows selected (Label=JA): {len(selected)}")

    return txt_paths

