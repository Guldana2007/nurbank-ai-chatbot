# build_faiss_bge3.py
import json
import argparse
import numpy as np
import faiss
import time
import os
from sentence_transformers import SentenceTransformer

# --------- Константы ----------
CHUNKED_FILE = "data/chunked_data.json"
INDEX_DIR = "embeddings_bge3"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.json")
EMB_FILE = os.path.join(INDEX_DIR, "embeddings.npy")

MODEL_NAME = "BAAI/bge-m3"
USE_GPU = False  # Поставь True, если есть CUDA
BATCH_SIZE = 8
# ------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def embed_passages(model, texts, normalize=True, batch_size=64):
    """Для bge-* желательно добавлять префикс 'passage: ' к документам."""
    prefixed = [f"passage: {t}" for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True
    )
    return embs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked", default=CHUNKED_FILE)
    parser.add_argument("--outdir", default=INDEX_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--use-gpu", action="store_true", default=USE_GPU)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    print(f"[1/4] Загружаем chunks из {args.chunked} ...")
    with open(args.chunked, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [d["content"] for d in docs]
    print(f"Всего фрагментов: {len(texts)}")

    print(f"[2/4] Загружаем модель эмбеддингов: {args.model}")
    start = time.time()
    model = SentenceTransformer(args.model, device="cuda" if args.use_gpu else "cpu")
    print(f"Модель загружена за {time.time() - start:.2f} сек.")

    print("[3/4] Считаем эмбеддинги (с нормализацией) ...")
    start = time.time()
    embeddings = embed_passages(model, texts, normalize=True, batch_size=args.batch_size)
    print(f"Эмбеддинги посчитаны за {time.time() - start:.2f} сек. Shape: {embeddings.shape}")

    print("[4/4] Создаём FAISS IndexFlatIP (cosine через нормализацию)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # с нормализованными векторами это == cosine similarity
    index.add(embeddings)

    print(f"Сохраняем индекс в {args.outdir}")
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
    meta = {
        "model": args.model,
        "dim": dim,
        "count": len(texts),
        "built_at": time.time()
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Готово!")

if __name__ == "__main__":
    main()
