# parse_urls_fast.py
import os
import json
import asyncio
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

INPUT_FILE = "data/urls.txt"
OUTPUT_FILE = "data/embeddings_input.jsonl"

def extract_text(soup):
    # Удаляем скрипты и стили
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    texts = []

    # Основной видимый текст
    for tag in soup.find_all(string=True):
        text = tag.strip()
        if text and len(text.split()) > 3:
            texts.append(text)

    # alt-тексты картинок
    for img in soup.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if alt and alt not in texts:
            texts.append(alt)

    return "\n".join(texts)

async def fetch_page(playwright, url):
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=20000)
        await page.wait_for_timeout(1000)
        html = await page.content()
    except Exception as e:
        print(f"[ОШИБКА] {url}: {e}")
        html = ""
    await browser.close()
    return html

async def main():
    os.makedirs("data", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    results = []

    async with async_playwright() as p:
        for url in tqdm(urls, desc="Сбор контента"):
            html = await fetch_page(p, url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            text = extract_text(soup)
            text = " ".join(text.split())
            if len(text) > 200:
                results.append({
                    "url": url,
                    "content": text
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Сохранено {len(results)} страниц в {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
