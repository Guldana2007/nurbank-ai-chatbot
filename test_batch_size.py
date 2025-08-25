import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
device = "cuda"

model = SentenceTransformer(MODEL_NAME, device=device)

def test_batch_size(batch_size):
    try:
        print(f"Пробуем batch_size = {batch_size} ...", end="")
        dummy_texts = ["passage: тест"] * batch_size
        model.encode(dummy_texts, normalize_embeddings=True, convert_to_numpy=True)
        print(" ✅ OK")
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(" ❌ CUDA OOM")
            torch.cuda.empty_cache()
            return False
        else:
            raise

def find_max_batch_size(start=1, end=64):
    best = start
    while start <= end:
        mid = (start + end) // 2
        if test_batch_size(mid):
            best = mid
            start = mid + 1
        else:
            end = mid - 1
    return best

if __name__ == "__main__":
    max_batch = find_max_batch_size()
    print(f"\n✅ Максимальный batch_size для {MODEL_NAME}: {max_batch}")
