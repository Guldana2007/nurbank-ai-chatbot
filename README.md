AI ChatBot для сайта Nurbank.kz

Этот проект представляет собой локальный AI-ассистент для сайта www.nurbank.kz
, который отвечает на вопросы пользователей на основе данных, автоматически собранных и обработанных из официального сайта банка. Система построена на RAG-архитектуре: парсинг контента, создание векторной базы знаний и генерация ответов локальной LLM-моделью. Поддерживаются три языка: русский, казахский и английский.

Основные возможности

– Поддержка трёх языков.
– Локальная база знаний (FAISS + BGE-M3).
– Генерация ответов моделью Llama3 через Ollama.
– Удобный чат-интерфейс (React + Vite).
– Отображение времени сообщений.
– Индикация «бот печатает».
– Светлая и тёмная темы.
– Полностью оффлайн-режим, без внешних API.

Структура проекта

ai_nur_bot_upgraded-nfactorial/
app/ – backend (FastAPI, Ollama, FAISS);
nurbank-ai-frontend/ – фронтенд (React + Vite);
data/ – подготовленные данные (urls, chunked_data, embeddings_input);
embeddings_bge3/ – FAISS индекс и эмбеддинги;
build_faiss_bge3.py – построение индекса;
parse_content_fast.py – парсинг HTML;
crawl_urls_sitemap.py – сбор URL;
requirements.txt;
README.md.

Установка и запуск

Клонирование репозитория:
git clone https://github.com/your-username/ai_nur_bot_upgraded-nfactorial.git

cd ai_nur_bot_upgraded-nfactorial

Backend (FastAPI + Ollama)
Установка зависимостей:
pip install -r requirements.txt
Запуск модели:
ollama run llama3
Запуск сервера:
cd app
uvicorn main_ollama_site_lang_bge3:app --port 9000 --reload
Проверка:
http://127.0.0.1:9000
 — «Hello! NurBank AI Assistant is running»
http://127.0.0.1:9000/health
 — { "status": "ok" }

Frontend (React + Vite)
cd nurbank-ai-frontend
npm install
npm run dev
Интерфейс доступен по адресу http://localhost:5173
.

Принцип работы

Парсер извлекает контент со всех страниц сайта Nurbank.kz.

Каждая страница превращается в отдельный семантический чанк.

Модель BGE-M3 генерирует эмбеддинги.

Эмбеддинги индексируются в FAISS.

При запросе пользователя выполняется поиск релевантных документов, формируется промпт и передаётся в Llama3.

Интерфейс отображает ответ в чате.

Технический стек

LLM — Llama3 через Ollama (локально, стабильно, бесплатно).
Embeddings — BAAI/BGE-M3 (точная работа с русским, казахским и английским).
Vector DB — FAISS (быстро, оффлайн).
Backend — FastAPI.
Frontend — React + Vite.

Уникальные решения

– Каждому URL соответствует один чанк — повышение точности.
– Полностью локальная система.
– Мультиязычная поддержка.
– Удобные элементы интерфейса.

Компромиссы

– Локальный backend требует публичного сервера для продакшена фронтенда.
– Динамические блоки сайта могут не парситься.
– Страницы раздела /cards/ частично дополнены вручную.

Известные проблемы

– Возможные дубли в chunked_data.json при ошибках парсинга.
– Дубли негативно влияют на качество индекса.
– Длинные ответы могут генерироваться до 10 секунд.

Деплой

Frontend можно развернуть на Vercel или Netlify.
Backend — на Render, Railway или собственном сервере.
Для фронтенда требуется указать адрес backend:
VITE_API_URL=https://ваш-бэкенд.onrender.com

Автор проекта

Guldana Kassym-Ashim
Team Lead, RPA & AI
АО «Нурбанк»
