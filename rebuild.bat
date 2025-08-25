@echo off
echo [1] Загружаем URL из sitemap...
python crawl_urls_sitemap.py

echo [2] Парсим страницы и создаем чанки...
python parse_content_fast.py

echo [3] Создаем FAISS-индекс с BGE-M3...
python build_faiss_bge3.py

echo [ГОТОВО] Всё успешно пересоздано!
pause
