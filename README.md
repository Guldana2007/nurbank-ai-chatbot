🤖 AI ChatBot для сайта www.nurbank.kz

Этот проект — AI-ассистент Нурбанка, который отвечает на вопросы пользователей на основе информации с сайта https://www.nurbank.kz
.
В проекте реализован чат с удобным интерфейсом, поддерживающий русский, казахский и английский языки.

🚀 Возможности

📌 Поддержка 3 языков: русский, казахский, английский

📚 Использование локальной базы знаний (FAISS + BGE3)

🧠 Генерация ответов локальной LLM-моделью (llama3 через Ollama)

💬 Удобный интерфейс чата (React + Vite)

⏰ Отображение времени сообщений

⌛ Лоадер «Бот печатает…»

🌙/☀️ Переключение между светлой и тёмной темами

🔒 Работа оффлайн (все модели локальные)

📁 Структура проекта
ai_nur_bot_upgraded-nfactorial/
├── app/                         # Backend (FastAPI + Ollama + FAISS)
│   ├── main_ollama_site_lang_bge3.py
│   ├── config.py
│   ├── retriever.py
│   └── llm.py
├── nurbank-ai-frontend/          # Frontend (React + Vite)
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── data/                        # Данные для embed'динга
│   ├── urls.txt
│   ├── chunked_data.json
│   └── embeddings_input.jsonl
├── embeddings_bge3/             # FAISS-индекс
│   ├── index.faiss
│   └── embeddings.npy
├── build_faiss_bge3.py          # Индексация векторной базы
├── parse_content_fast.py        # Парсинг HTML с сайта
├── crawl_urls_sitemap.py        # Сбор URL по sitemap
├── requirements.txt             # Python-зависимости
└── README.md

⚙️ Установка и запуск
1. Клонирование репозитория
git clone https://github.com/your-username/ai_nur_bot_upgraded-nfactorial.git
cd ai_nur_bot_upgraded-nfactorial

2. Backend (FastAPI + Ollama)

Установите зависимости:

pip install -r requirements.txt


Запустите Ollama и модель:

ollama run llama3


Перейдите в папку backend и запустите сервер:

cd app
uvicorn main_ollama_site_lang_bge3:app --port 9000 --reload


Проверьте:

http://127.0.0.1:9000
 → должно быть сообщение "Hello! NurBank AI Assistant is running 🚀"

http://127.0.0.1:9000/health
 → {"status": "ok"}

3. Frontend (React + Vite)

Перейдите в папку фронтенда:

cd nurbank-ai-frontend
npm install
npm run dev


Откройте: http://localhost:5173

🧠 Как работает

Сайт Нурбанка парсится (parse_content_fast.py)

Каждая страница = 1 чанк (сохранение семантики)

Чанки обрабатываются моделью BAAI/bge-m3 и индексируются через FAISS

При запросе пользователя:

система ищет релевантные документы в FAISS

формирует промпт

передаёт в LLM (llama3 через Ollama)

возвращает готовый ответ

Интерфейс (React) отображает ответ в чате

💡 Почему выбран этот стек
Компонент	Выбор	Обоснование
LLM	llama3 (через Ollama)	Локальная, бесплатная, работает оффлайн
Embeddings	BAAI/bge-m3	Поддержка многоязычия, высокая точность
Векторная БД	FAISS	Быстрая и лёгкая в работе
Backend	FastAPI	Быстрый Python-фреймворк для API
Frontend	React + Vite	Удобный, быстрый, современный UI
🎯 Уникальные подходы

1 URL = 1 чанк — сохранение целостности текста

Полностью оффлайн — база знаний и LLM работают локально

Мультиязычность — поддержка русского, казахского, английского

UI-фичи: время сообщений, лоадер «бот печатает…», светлая/тёмная тема

🤝 Компромиссы

Backend пока локальный → фронтенд на Vercel не сможет достучаться, если не задеплоить backend (например, на Render).

Парсинг сайта мог пропустить скрытый контент (FAQ, динамические блоки).

Некоторые страницы /cards/ были дополнены вручную.

🐞 Известные проблемы

Иногда дубли в chunked_data.json при сбое парсинга

Если не фильтровать дубли, FAISS индекс засоряется

При долгих ответах LLM возможна задержка (до 10 сек)

🌐 Деплой

Frontend можно развернуть на Vercel/Netlify

Backend — на Render/Railway или собственном сервере

В .env у фронтенда нужно прописать адрес backend:

VITE_API_URL=https://ваш-бэкенд.onrender.com

🧠 Автор проекта

Guldana Kassym-Ashim
RPA and AI Team Lead, АО «Нурбанк»


