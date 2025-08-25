# parse_content_fast.py
import json
import os
from tqdm import tqdm

INPUT_FILE = "data/embeddings_input.jsonl"
OUTPUT_FILE = "data/chunked_data.json"

def normalize_text(text):
    return " ".join(text.strip().split())

def load_input_jsonl(path):
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("content", "").strip():
                    pages.append(item)
            except json.JSONDecodeError:
                continue
    return pages

def main():
    print(f"[1/3] Загружаем данные из {INPUT_FILE} ...")
    pages = load_input_jsonl(INPUT_FILE)
    print(f"[2/3] Обрабатываем {len(pages)} страниц (1 URL = 1 чанк)...")

    chunks = []
    for page in tqdm(pages):
        url = page.get("url", "")
        content = normalize_text(page.get("content", ""))

        chunks.append({
            "url": url,
            "content": content
        })

    print(f"[3/3] Сохраняем {len(chunks)} чанков в {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Готово!")

if __name__ == "__main__":
    main()
