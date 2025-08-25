import requests
from bs4 import BeautifulSoup

url = "https://nurbank.kz/ru/person/cards/visa-gold/"
r = requests.get(url, timeout=5)
soup = BeautifulSoup(r.text, "lxml")
for s in soup(['script', 'style']):
    s.extract()

text = soup.get_text(separator=" ", strip=True)
print(f"Длина текста: {len(text)} символов")
print("--- Текст ---")
print(text[:2000])  # первые 2000 символов
