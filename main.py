import asyncio
import logging
import os
import sqlite3
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Set, Optional
from contextlib import contextmanager

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

# ==================== КОНФИГУРАЦИЯ ====================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"
FALLBACK_OPENROUTER_MODEL = "z-ai/glm-4.5-air"
OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", DEFAULT_OPENROUTER_MODEL).strip().strip("\"'")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()

ACK_TEXT = os.getenv("BOT_ACK_TEXT", "Запрос принят, обрабатываю…").strip() or "Запрос принят, обрабатываю…"
BUSY_TEXT = os.getenv("BOT_BUSY_TEXT", "Подожди, сейчас обрабатываю твоё прошлое сообщение.").strip() or "Подожди, сейчас обрабатываю твоё прошлое сообщение."
USER_ERROR_AI = os.getenv("BOT_USER_ERROR_TEXT", "Не удалось получить ответ. Попробуй позже.").strip() or "Не удалось получить ответ. Попробуй позже."

MAX_HISTORY_MESSAGES = 20
DB_PATH = "bot.db"

BOT_STARTED_AT = time.time()

# ==================== БАЗА ДАННЫХ ====================

def init_db():
    """Инициализация базы данных — создает таблицы если их нет"""
    with sqlite3.connect(DB_PATH) as conn:
        # Таблица истории диалогов
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица статистики (ключ-значение)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
        """)
        
        # Индексы для быстрого поиска
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp)")
        
        # Инициализируем счетчики статистики, если их нет
        conn.execute("INSERT OR IGNORE INTO stats (key, value) VALUES ('users_count', 0)")
        conn.execute("INSERT OR IGNORE INTO stats (key, value) VALUES ('ai_requests', 0)")
        
        conn.commit()

@contextmanager
def get_db():
    """Контекстный менеджер для работы с БД"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def save_message(user_id: int, role: str, content: str):
    """Сохранить сообщение в историю"""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )

def get_history_from_db(user_id: int, limit: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """Получить историю диалога пользователя из БД"""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cur.fetchall()
        # Возвращаем в хронологическом порядке (от старых к новым)
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def clear_history_in_db(user_id: int):
    """Очистить историю диалога пользователя"""
    with get_db() as conn:
        conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))

def get_stats_from_db() -> tuple[int, int]:
    """Получить статистику из БД (количество пользователей, количество запросов)"""
    with get_db() as conn:
        cur = conn.execute("SELECT value FROM stats WHERE key = 'users_count'")
        users_count = cur.fetchone()[0]
        cur = conn.execute("SELECT value FROM stats WHERE key = 'ai_requests'")
        requests_count = cur.fetchone()[0]
    return users_count, requests_count

def increment_users_count():
    """Увеличить счетчик уникальных пользователей"""
    with get_db() as conn:
        conn.execute(
            "UPDATE stats SET value = value + 1 WHERE key = 'users_count' AND value = (SELECT value FROM stats WHERE key = 'users_count')"
        )

def increment_ai_requests():
    """Увеличить счетчик запросов к AI"""
    with get_db() as conn:
        conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'ai_requests'")

def get_or_create_user_history(user_id: int) -> List[Dict[str, str]]:
    """Получить историю пользователя из БД или вернуть пустой список"""
    history = get_history_from_db(user_id, MAX_HISTORY_MESSAGES)
    return history

def sync_history_to_db(user_id: int, history: Deque[Dict[str, str]]):
    """Синхронизировать историю из памяти в БД (сохраняет только последние сообщения)"""
    # Очищаем старую историю
    clear_history_in_db(user_id)
    # Сохраняем новые сообщения
    for msg in history:
        save_message(user_id, msg["role"], msg["content"])

# ==================== КЭШ В ПАМЯТИ ====================

# user_id -> deque of messages (кэш для быстрого доступа)
chat_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))

# Пользователи, у которых сейчас выполняется запрос к ИИ
processing_user_ids: Set[int] = set()

def load_all_histories_to_cache():
    """Загрузить всю историю из БД в кэш при старте бота"""
    with get_db() as conn:
        cur = conn.execute("SELECT DISTINCT user_id FROM history")
        user_ids = [row[0] for row in cur.fetchall()]
        
        for user_id in user_ids:
            history = get_history_from_db(user_id, MAX_HISTORY_MESSAGES)
            chat_history[user_id] = deque(history, maxlen=MAX_HISTORY_MESSAGES)
    
    logging.info(f"Загружена история для {len(chat_history)} пользователей")

def register_user(user_id: int) -> bool:
    """Зарегистрировать нового пользователя. Возвращает True если пользователь новый"""
    with get_db() as conn:
        # Проверяем, есть ли у пользователя сообщения в истории
        cur = conn.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
        count = cur.fetchone()[0]
        
        if count == 0:
            # Новый пользователь
            conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'users_count'")
            return True
    return False

# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С OPENROUTER ====================

def validate_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")

async def ask_openrouter(
    session: aiohttp.ClientSession, user_id: int, user_text: str
) -> str:
    # Сохраняем сообщение пользователя в БД
    save_message(user_id, "user", user_text)
    
    # Добавляем в кэш
    chat_history[user_id].append({"role": "user", "content": user_text})
    
    # Формируем сообщения для API (используем кэш)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful Telegram assistant. "
                "Reply briefly and clearly in the same language as the user."
            ),
        },
        *list(chat_history[user_id]),
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    endpoint = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    
    try:
        async with session.post(endpoint, json=payload, headers=headers, timeout=60) as resp:
            if resp.status == 404 and payload["model"] != FALLBACK_OPENROUTER_MODEL:
                logging.warning(
                    "OpenRouter model '%s' not found. Falling back to '%s'.",
                    payload["model"],
                    FALLBACK_OPENROUTER_MODEL,
                )
                payload["model"] = FALLBACK_OPENROUTER_MODEL
                async with session.post(endpoint, json=payload, headers=headers, timeout=60) as retry_resp:
                    if retry_resp.status != 200:
                        err_body = await retry_resp.text()
                        logging.error("LLM API error after fallback (status=%s): %s", retry_resp.status, err_body)
                        raise RuntimeError(USER_ERROR_AI)
                    data = await retry_resp.json()
            elif resp.status != 200:
                error_text = await resp.text()
                logging.warning("LLM API error (status=%s): %s", resp.status, error_text)
                
                # Обработка специфических ошибок
                if resp.status == 429:
                    return "⚠️ Слишком много запросов. Подождите немного."
                if resp.status == 402:
                    return "💰 Закончились кредиты OpenRouter. Уведомите администратора."
                if resp.status == 401:
                    return "🔑 Проблема с API-ключом. Уведомите администратора."
                
                raise RuntimeError(USER_ERROR_AI)
            else:
                data = await resp.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Извини, не смог сгенерировать ответ.")
        )
        
        # Сохраняем ответ ассистента в БД
        save_message(user_id, "assistant", answer)
        
        # Добавляем в кэш
        chat_history[user_id].append({"role": "assistant", "content": answer})
        
        return answer
        
    except asyncio.TimeoutError:
        logging.error("OpenRouter request timeout")
        return "⏰ Превышено время ожидания ответа. Попробуйте позже."
    except aiohttp.ClientError as e:
        logging.error(f"Network error: {e}")
        return "🌐 Не удалось подключиться к AI. Проверьте интернет."

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

async def main() -> None:
    # Инициализация
    validate_env()
    init_db()  # Создаем таблицы
    load_all_histories_to_cache()  # Загружаем историю из БД в кэш
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Настройка прокси
    tg_session = AiohttpSession(proxy=TELEGRAM_PROXY_URL or None)
    bot = Bot(token=BOT_TOKEN, session=tg_session)
    dp = Dispatcher()
    session = aiohttp.ClientSession()
    
    # ==================== КОМАНДЫ ====================
    
    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        if message.from_user:
            is_new = register_user(message.from_user.id)
            if is_new:
                logging.info(f"Новый пользователь: {message.from_user.id}")
        await message.answer(
            "🤖 Добро пожаловать в PrimeAi бота!\n\n"
            "Задай мне любой вопрос — я отвечу с помощью ИИ.\n"
            "Я помню историю нашего диалога, даже после перезапуска!\n\n"
            "📌 Команды:\n"
            "/clear — очистить историю диалога\n"
            "/stats — моя статистика\n"
            "/help — подробная справка"
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        if message.from_user:
            register_user(message.from_user.id)
        await message.answer(
            "<b>🤖 Помощь — PrimeAi</b>\n\n"
            "<b>💬 Чат с ИИ</b>\n"
            "Просто напиши любое текстовое сообщение — я отвечу с помощью нейросети.\n"
            "Я помню последние 20 сообщений нашего диалога.\n\n"
            "<b>📋 Команды</b>\n"
            "/start — приветствие\n"
            "/help — эта справка\n"
            "/clear — очистить историю диалога\n"
            "/stats — статистика работы бота\n"
            "/whoami — твой Telegram ID\n\n"
            "<b>👑 Админ-команды</b>\n"
            "/stats — полная статистика\n"
            "/admin_clear &lt;user_id&gt; — очистить историю пользователя",
            parse_mode="HTML",
        )

    @dp.message(Command("whoami"))
    async def cmd_whoami(message: Message) -> None:
        uid = message.from_user.id if message.from_user else None
        await message.answer(
            f"🆔 Ваш Telegram ID: `{uid}`\n\n"
            "Его можно указать в ADMIN_IDS в .env для доступа к админ-командам.\n"
            "Пример: ADMIN_IDS=123456789,987654321",
            parse_mode="Markdown"
        )

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        user_id = message.from_user.id
        chat_history[user_id].clear()
        clear_history_in_db(user_id)
        await message.answer("🧹 История диалога очищена!")

    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        users_count, requests_count = get_stats_from_db()
        uptime_sec = int(time.time() - BOT_STARTED_AT)
        h, rem = divmod(uptime_sec, 3600)
        m, s = divmod(rem, 60)
        
        # Проверяем, админ ли пользователь
        from dotenv import load_dotenv
        load_dotenv()
        admin_ids_str = os.getenv("ADMIN_IDS", "")
        admin_ids = {int(x.strip()) for x in admin_ids_str.split(",") if x.strip().isdigit()}
        is_admin_user = message.from_user and message.from_user.id in admin_ids
        
        if is_admin_user:
            await message.answer(
                f"📊 *Статистика PrimeAi*\n\n"
                f"👥 *Уникальных пользователей:* {users_count}\n"
                f"🤖 *Запросов к ИИ:* {requests_count}\n"
                f"⏱️ *Аптайм:* {h}ч {m}м {s}с\n"
                f"🧠 *Модель:* `{OPENROUTER_MODEL}`\n"
                f"📝 *max_tokens:* {OPENROUTER_MAX_TOKENS}\n"
                f"💾 *Хранилище:* SQLite (данные сохраняются)",
                parse_mode="Markdown"
            )
        else:
            await message.answer(
                f"🤖 *PrimeAi — статистика*\n\n"
                f"🧠 Модель: `{OPENROUTER_MODEL}`\n"
                f"📝 Максимальная длина ответа: {OPENROUTER_MAX_TOKENS} символов\n"
                f"💬 Я помню последние {MAX_HISTORY_MESSAGES} сообщений\n"
                f"⏱️ Работаю без перерыва: {h}ч {m}м {s}с\n\n"
                f"💡 *Совет:* Используй /clear чтобы начать диалог заново!",
                parse_mode="Markdown"
            )

    @dp.message(Command("admin"))
    async def cmd_admin(message: Message) -> None:
        if not message.from_user:
            return
        
        admin_ids_str = os.getenv("ADMIN_IDS", "")
        admin_ids = {int(x.strip()) for x in admin_ids_str.split(",") if x.strip().isdigit()}
        
        if not admin_ids:
            await message.answer(
                "🔧 Админка не настроена.\n"
                "Добавь ADMIN_IDS в .env файл.\n"
                "Пример: ADMIN_IDS=123456789"
            )
            return
        
        if message.from_user.id not in admin_ids:
            await message.answer("⛔ Нет доступа к админ-командам.")
            return
        
        await message.answer(
            "👑 *Админ-панель PrimeAi*\n\n"
            "Доступные команды:\n"
            "/stats — полная статистика\n"
            "/admin_clear <user_id> — очистить историю пользователя\n"
            "/whoami — показать свой Telegram ID\n\n"
            "Все данные сохраняются в SQLite и не теряются при перезапуске!",
            parse_mode="Markdown"
        )

    @dp.message(Command("admin_clear"))
    async def cmd_admin_clear(message: Message, command: CommandObject) -> None:
        if not message.from_user:
            return
        
        admin_ids_str = os.getenv("ADMIN_IDS", "")
        admin_ids = {int(x.strip()) for x in admin_ids_str.split(",") if x.strip().isdigit()}
        
        if not admin_ids:
            await message.answer("ADMIN_IDS не задан в .env.")
            return
        
        if message.from_user.id not in admin_ids:
            await message.answer("⛔ Нет доступа.")
            return
        
        args = (command.args or "").strip()
        if not args:
            await message.answer("Использование: /admin_clear <user_id>")
            return
        if not args.isdigit():
            await message.answer("user_id должен быть числом.")
            return
        
        uid = int(args)
        chat_history[uid].clear()
        clear_history_in_db(uid)
        await message.answer(f"✅ История диалога для user_id={uid} очищена.")

    @dp.message(F.text)
    async def handle_text(message: Message) -> None:
        user = message.from_user
        if user is None or message.text is None:
            return

        if user.id in processing_user_ids:
            await message.answer(BUSY_TEXT)
            return

        processing_user_ids.add(user.id)
        try:
            # Регистрируем пользователя (увеличиваем счетчик если новый)
            register_user(user.id)
            increment_ai_requests()
            
            ack_msg = await message.answer(ACK_TEXT)
            try:
                await bot.send_chat_action(
                    chat_id=message.chat.id,
                    action=ChatAction.TYPING,
                )
                reply = await ask_openrouter(session, user.id, message.text)
                
                # Разбиваем длинный ответ на части
                if len(reply) > 4000:
                    for i in range(0, len(reply), 4000):
                        await message.answer(reply[i:i+4000])
                else:
                    await message.answer(reply)
                    
            except Exception as e:
                logging.exception("Failed to handle message")
                await message.answer(USER_ERROR_AI)
            finally:
                try:
                    await bot.delete_message(
                        chat_id=ack_msg.chat.id,
                        message_id=ack_msg.message_id,
                    )
                except Exception:
                    logging.debug("Could not delete ack message", exc_info=True)
        finally:
            processing_user_ids.discard(user.id)

    # Проверка подключения к Telegram
    connected = False
    for attempt in range(1, 4):
        try:
            me = await bot.get_me()
            logging.info("Telegram connected as @%s", me.username)
            connected = True
            break
        except Exception as exc:
            logging.warning("Telegram connection attempt %s/3 failed: %s", attempt, exc)
            await asyncio.sleep(2 * attempt)

    if not connected:
        raise RuntimeError(
            "Cannot connect to Telegram API. "
            "Check internet, bot token, firewall, or set TELEGRAM_PROXY_URL in .env."
        )

    logging.info(f"✅ Бот PrimeAi запущен! Модель: {OPENROUTER_MODEL}")
    
    try:
        await dp.start_polling(bot)
    finally:
        await session.close()
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())