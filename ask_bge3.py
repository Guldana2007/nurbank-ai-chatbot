# ask_bge3.py
import requests

API_URL = "http://localhost:9000/ask"

def main():
    print("Введите вопрос (или 'exit' для выхода):")
    while True:
        question = input("> ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        try:
            response = requests.post(API_URL, json={"question": question})
            data = response.json()
            print("\n[ОТВЕТ]:", data.get("answer", "[нет ответа]"))
            if "used_urls" in data:
                print("\n[ИСТОЧНИК]:")
                for url in data["used_urls"]:
                    print(" -", url)
            print("-" * 50)
        except Exception as e:
            print("Ошибка запроса:", e)

if __name__ == "__main__":
    main()

