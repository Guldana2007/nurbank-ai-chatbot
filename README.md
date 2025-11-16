AI ChatBot for Nurbank.kz — RAG + Local LLM Pipeline

Этот проект реализует полноценный AI-ассистент для сайта www.nurbank.kz
, который отвечает на вопросы пользователей строго на основе информации, полученной с официального сайта банка.

Система сочетает локальную базу знаний (FAISS + BGE-M3), парсер HTML-контента, LLM (Llama3 через Ollama) и современный UI (React + Vite).
Поддерживаются три языка: русский, казахский и английский.

Проект демонстрирует навыки в:

разработке RAG-архитектур

сборе и обработке данных

векторном поиске

работе с локальными LLM

проектировании фронтенд и бекенд систем

инженерии CI/CD и деплоя

Key Capabilities

 Поддержка RU / KZ / EN

Локальная база знаний (FAISS + BGE3)

Генерация ответов моделью Llama3 (Ollama)

UI чата (React + Vite)

Время сообщений

Лоадер «бот печатает…»

Светлая/тёмная тема

Полностью оффлайн-режим

Project Structure
ai_nur_bot_upgraded-nfactorial/
├── app/                         # Backend (FastAPI + Ollama + FAISS)
│   ├── main_ollama_site_lang_bge3.py
│   ├── config.py
│   ├── retriever.py
│   └── llm.py
│
├── nurbank-ai-frontend/         # Frontend (React + Vite)
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/                        # Данные для embed'динга
│   ├── urls.txt
│   ├── chunked_data.json
│   └── embeddings_input.jsonl
│
├── embeddings_bge3/             # FAISS-индекс
│   ├── index.faiss
│   └── embeddings.npy
│
├── build_faiss_bge3.py          # Генерация FAISS индекса
├── parse_content_fast.py         # Парсинг HTML контента
├── crawl_urls_sitemap.py         # Сбор URL через sitemap
├── requirements.txt
└── README.md

System Architecture
                +-------------------------------+
                |   Website: www.nurbank.kz     |
                +-------------------------------+
                              |
                              v
                +-------------------------------+
                |   parse_content_fast.py       |
                |  (1 URL = 1 semantic chunk)   |
                +-------------------------------+
                              |
                              v
                  +-----------------------+
                  | embeddings_input.jsonl|
                  +-----------------------+
                              |
                              v
                +-------------------------------+
                |      build_faiss_bge3.py      |
                | - BGE-M3 embeddings           |
                | - FAISS index                 |
                +-------------------------------+
                              |
                    +---------+---------+
                    |                   |
                    v                   v
         +------------------+   +---------------------+
         |  embeddings.npy  |   |    index.faiss      |
         +------------------+   +---------------------+
                    ^                   |
                    |                   v
                +-------------------------------+
                |         retriever.py          |
                +-------------------------------+
                              |
                              v
                +-------------------------------+
                |    Llama3 (via Ollama API)    |
                +-------------------------------+
                              |
                              v
                +-------------------------------+
                |     FastAPI backend (app/)    |
                +-------------------------------+
                              |
                              v
                +-------------------------------+
                |     React + Vite frontend     |
                +-------------------------------+

Installation & Run
1. Clone
git clone https://github.com/your-username/ai_nur_bot_upgraded-nfactorial.git
cd ai_nur_bot_upgraded-nfactorial

2. Backend (FastAPI + Ollama)

Установка зависимостей:

pip install -r requirements.txt


Запуск модели:

ollama run llama3


Запуск сервера:

cd app
uvicorn main_ollama_site_lang_bge3:app --port 9000 --reload


Проверка:

http://127.0.0.1:9000
 → “Hello! NurBank AI Assistant is running 🚀”

http://127.0.0.1:9000/health
 → { "status": "ok" }

3. Frontend (React + Vite)
cd nurbank-ai-frontend
npm install
npm run dev


Открыть: http://localhost:5173

How It Works

Скрипт parse_content_fast.py собирает контент с nurbank.kz

Каждая страница → один семантический чанк

BGE-M3 создаёт эмбеддинги

FAISS индексирует документы

При запросе пользователя:

выполняется поиск по индексу

формируется промпт

Llama3 генерирует ответ

Интерфейс React отображает диалог

Why This Tech Stack
Компонент	Выбор	Обоснование
LLM	Llama3 + Ollama	Локальная, бесплатная, оффлайн
Embeddings	BGE-M3	Отличная точность в RU/KZ/EN
Vector DB	FAISS	Быстрый и лёгкий
Backend	FastAPI	Идеален для ML API
Frontend	React + Vite	Современный, быстрый UI
Unique Design Decisions

1 URL = 1 чанк

Полностью локальная работа (LLM + FAISS)

Мультиязычная генерация

UI: время сообщений, механизм «бот печатает…», темизация

Compromises

Backend локальный → деплой фронтенда требует публичного backend

FAQ и динамические блоки сайта могли быть пропущены

Некоторые страницы /cards/ дополнялись вручную

Known Issues

Возможны дубли в chunked_data.json

FAISS индекс может засоряться при ошибках парсинга

Длинные ответы LLM занимают до 5–10 сек

Deployment

Frontend можно разместить на:

Vercel

Netlify

Backend:

Render

Railway

VPS/сервер банка

Фронтенд переменная:

VITE_API_URL=https://ваш-бэкенд.onrender.com

Project Ownership

Author:
Guldana Kassym-Ashim
Team Lead, RPA & AI
АО «Нурбанк»
