# crawl_urls_sitemap.py
import requests
import xml.etree.ElementTree as ET
import os
import re
import argparse

DEFAULT_SITEMAP = "https://nurbank.kz/sitemap.xml"
DEFAULT_OUTPUT_FILE = "data/urls.txt"

def remove_namespace(xml):
    return re.sub(r' xmlns="[^"]+"', '', xml, count=1)

def fetch_urls_from_sitemap(sitemap_url):
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        xml_content = remove_namespace(response.text)

        root = ET.fromstring(xml_content)
        for url in root.findall(".//url/loc"):
            u = url.text.strip()
            if "press-center" in u or "news" in u:
                continue
            urls.append(u)
    except Exception as e:
        print(f"Ошибка при загрузке sitemap: {e}")
    return urls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitemap", type=str, default=DEFAULT_SITEMAP, help="URL sitemap-файла")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_FILE, help="Путь для сохранения URL")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    urls = fetch_urls_from_sitemap(args.sitemap)
    with open(args.out, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Собрано {len(urls)} URL из sitemap (без пресс-центра/новостей) и сохранено в {args.out}")
