import asyncio
import logging
import os
import sqlite3
import time
import json
import base64
from collections import defaultdict, deque
from typing import Deque, Dict, List, Set
from contextlib import contextmanager
from functools import wraps

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, BufferedInputFile
from dotenv import load_dotenv

load_dotenv()

# ==================== КОНФИГУРАЦИЯ ====================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v3.2"
OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", DEFAULT_OPENROUTER_MODEL).strip().strip("\"'")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()

# Модель для генерации изображений (можно переопределить в .env)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "openrouter/hunter-alpha").strip()

ACK_TEXT = os.getenv("BOT_ACK_TEXT", "Запрос принят, обрабатываю…").strip() or "Запрос принят, обрабатываю…"
BUSY_TEXT = os.getenv("BOT_BUSY_TEXT", "Подожди, сейчас обрабатываю твоё прошлое сообщение.").strip() or "Подожди, сейчас обрабатываю твоё прошлое сообщение."
USER_ERROR_AI = os.getenv("BOT_USER_ERROR_TEXT", "Не удалось получить ответ. Попробуй позже.").strip() or "Не удалось получить ответ. Попробуй позже."

MAX_HISTORY_MESSAGES = 20
DB_PATH = "bot.db"

# ==================== РЕЖИМЫ ОТВЕТА ====================

MODES = {
    "brief": {
        "name": "📝 Краткий",
        "emoji": "📝",
        "system": "Отвечай максимально кратко, только суть. 1-2 предложения. Без лишних слов."
    },
    "balanced": {
        "name": "⚖️ Сбалансированный",
        "emoji": "⚖️",
        "system": "Отвечай кратко и понятно, но с достаточными деталями. 2-4 предложения."
    },
    "detailed": {
        "name": "📚 Подробный",
        "emoji": "📚",
        "system": "Отвечай подробно, объясняя все детали. Используй примеры, если уместно. 4-8 предложений."
    },
    "creative": {
        "name": "🎨 Креативный",
        "emoji": "🎨",
        "system": "Отвечай творчески, используй метафоры, образные выражения и юмор. Будь вдохновляющим."
    },
    "technical": {
        "name": "💻 Технический",
        "emoji": "💻",
        "system": "Отвечай технически точно, используй правильную терминологию. Будь конкретным и структурированным."
    }
}

DEFAULT_MODE = "balanced"

BOT_STARTED_AT = time.time()

# ==================== АДМИНКА ====================

def _parse_admin_ids() -> Set[int]:
    raw = os.getenv("ADMIN_IDS", "").strip()
    if not raw:
        return set()
    ids: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids

ADMIN_IDS = _parse_admin_ids()

def is_admin(user_id: int | None) -> bool:
    if user_id is None or not ADMIN_IDS:
        return False
    return user_id in ADMIN_IDS

def admin_only(func):
    @wraps(func)
    async def wrapper(message: Message, *args, **kwargs):
        if not message.from_user:
            return
        if not is_admin(message.from_user.id):
            await message.answer(
                "⛔ *Доступ запрещен*\n\n"
                "Эта команда доступна только администраторам бота.\n"
                "Узнать свой ID можно командой `/whoami`.",
                parse_mode="Markdown"
            )
            return
        return await func(message, *args, **kwargs)
    return wrapper

# ==================== БАЗА ДАННЫХ ====================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                mode TEXT DEFAULT 'balanced',
                lang TEXT DEFAULT 'ru',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp)")
        conn.execute("INSERT OR IGNORE INTO stats (key, value) VALUES ('users_count', 0)")
        conn.execute("INSERT OR IGNORE INTO stats (key, value) VALUES ('ai_requests', 0)")
        conn.commit()

@contextmanager
def get_db():
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
    with get_db() as conn:
        conn.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )

def get_history_from_db(user_id: int, limit: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def clear_history_in_db(user_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))

def get_stats_from_db() -> tuple[int, int]:
    with get_db() as conn:
        cur = conn.execute("SELECT value FROM stats WHERE key = 'users_count'")
        users_count = cur.fetchone()[0]
        cur = conn.execute("SELECT value FROM stats WHERE key = 'ai_requests'")
        requests_count = cur.fetchone()[0]
    return users_count, requests_count

def increment_users_count():
    with get_db() as conn:
        conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'users_count'")

def increment_ai_requests():
    with get_db() as conn:
        conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'ai_requests'")

def register_user(user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
        count = cur.fetchone()[0]
        if count == 0:
            conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'users_count'")
            return True
    return False

def save_user_mode(user_id: int, mode: str):
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_settings (user_id, mode)
                VALUES (?, ?)
            """, (user_id, mode))
    except Exception as e:
        logging.error(f"Ошибка сохранения режима: {e}")

def load_user_modes_from_db():
    try:
        with get_db() as conn:
            cur = conn.execute("SELECT user_id, mode FROM user_settings")
            rows = cur.fetchall()
            for user_id, mode in rows:
                if mode in MODES:
                    user_mode[user_id] = mode
            logging.info(f"Загружены настройки для {len(rows)} пользователей")
    except Exception as e:
        logging.error(f"Ошибка загрузки настроек: {e}")

# ==================== КЭШ В ПАМЯТИ ====================

chat_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
processing_user_ids: Set[int] = set()
user_mode: Dict[int, str] = defaultdict(lambda: DEFAULT_MODE)

def load_all_histories_to_cache():
    with get_db() as conn:
        cur = conn.execute("SELECT DISTINCT user_id FROM history")
        user_ids = [row[0] for row in cur.fetchall()]
        for user_id in user_ids:
            history = get_history_from_db(user_id, MAX_HISTORY_MESSAGES)
            chat_history[user_id] = deque(history, maxlen=MAX_HISTORY_MESSAGES)
    logging.info(f"Загружена история для {len(chat_history)} пользователей")

# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С OPENROUTER ====================

def validate_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")

async def ask_openrouter(
    session: aiohttp.ClientSession, user_id: int, user_text: str
) -> str:
    save_message(user_id, "user", user_text)
    chat_history[user_id].append({"role": "user", "content": user_text})
    mode = user_mode.get(user_id, DEFAULT_MODE)
    system_prompt = MODES[mode]["system"]
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "You are a helpful Telegram assistant. "
                "Reply in the same language as the user."
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
            if resp.status != 200:
                error_text = await resp.text()
                logging.warning("LLM API error (status=%s): %s", resp.status, error_text)
                if resp.status == 404:
                    return "❌ Модель временно недоступна. Попробуйте позже."
                if resp.status == 429:
                    return "⚠️ Слишком много запросов. Подождите немного."
                if resp.status == 402:
                    return "💰 Закончились кредиты. Уведомите администратора."
                if resp.status == 401:
                    return "🔑 Проблема с API-ключом. Уведомите администратора."
                raise RuntimeError(USER_ERROR_AI)
            data = await resp.json()
        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Извини, не смог сгенерировать ответ.")
        )
        save_message(user_id, "assistant", answer)
        chat_history[user_id].append({"role": "assistant", "content": answer})
        return answer
    except asyncio.TimeoutError:
        logging.error("OpenRouter request timeout")
        return "⏰ Превышено время ожидания ответа. Попробуйте позже."
    except aiohttp.ClientError as e:
        logging.error(f"Network error: {e}")
        return "🌐 Не удалось подключиться к AI. Проверьте интернет."

# ==================== КОМАНДЫ БОТА ====================

async def main() -> None:
    validate_env()
    init_db()
    load_all_histories_to_cache()
    load_user_modes_from_db()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tg_session = AiohttpSession(proxy=TELEGRAM_PROXY_URL or None)
    bot = Bot(token=BOT_TOKEN, session=tg_session)
    dp = Dispatcher()
    session = aiohttp.ClientSession()
    
    # ==================== КОМАНДЫ ДЛЯ ВСЕХ ====================
    
    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        if message.from_user:
            register_user(message.from_user.id)
        await message.answer(
            "🤖 *Добро пожаловать в PrimeAi бота!*\n\n"
            "Задай мне любой вопрос — я отвечу с помощью ИИ.\n"
            "Я помню историю нашего диалога, даже после перезапуска!\n\n"
            "📌 *Команды:*\n"
            "/clear — очистить историю\n"
            "/mode — выбрать стиль ответов\n"
            "/image — сгенерировать изображение\n"
            "/help — подробная справка\n"
            "/whoami — твой Telegram ID",
            parse_mode="Markdown"
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        if message.from_user:
            register_user(message.from_user.id)
        current_mode = user_mode.get(message.from_user.id, DEFAULT_MODE)
        current_mode_name = MODES[current_mode]["name"]
        help_text = (
            "<b>🤖 Помощь — PrimeAi</b>\n\n"
            "<b>💬 Чат с ИИ</b>\n"
            "Просто напиши любое текстовое сообщение — я отвечу с помощью нейросети.\n"
            "Я помню последние 20 сообщений нашего диалога.\n\n"
            "<b>📋 Команды</b>\n"
            "/start — приветствие\n"
            "/help — эта справка\n"
            "/clear — очистить историю диалога\n"
            "/mode — выбрать стиль ответов\n"
            "/image — сгенерировать изображение по описанию\n"
            "/whoami — твой Telegram ID\n\n"
            f"🎨 *Текущий режим:* {current_mode_name}\n\n"
        )
        if is_admin(message.from_user.id if message.from_user else None):
            help_text += (
                "<b>👑 Админ-команды</b>\n"
                "/stats — полная статистика бота\n"
                "/admin_clear &lt;user_id&gt; — очистить историю пользователя\n"
                "/admin — информация об админ-панели\n"
            )
        await message.answer(help_text, parse_mode="HTML")

    @dp.message(Command("whoami"))
    async def cmd_whoami(message: Message) -> None:
        uid = message.from_user.id if message.from_user else None
        is_admin_user = is_admin(uid)
        current_mode = user_mode.get(uid, DEFAULT_MODE)
        current_mode_name = MODES[current_mode]["name"]
        await message.answer(
            f"🆔 Ваш Telegram ID: `{uid}`\n\n"
            f"👑 Администратор: {'✅ Да' if is_admin_user else '❌ Нет'}\n"
            f"🎨 Режим ответа: {MODES[current_mode]['emoji']} *{current_mode_name}*\n\n"
            "Чтобы получить доступ к админ-командам, добавь этот ID в ADMIN_IDS в .env\n"
            "Пример: ADMIN_IDS=123456789",
            parse_mode="Markdown"
        )

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        user_id = message.from_user.id
        chat_history[user_id].clear()
        clear_history_in_db(user_id)
        await message.answer("🧹 История диалога очищена!")

    @dp.message(Command("mode"))
    async def cmd_mode(message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip().lower()
        if not args:
            mode_list = "\n".join([
                f"• `{code}` — {info['emoji']} {info['name']}" 
                for code, info in MODES.items()
            ])
            current_mode = user_mode.get(message.from_user.id, DEFAULT_MODE)
            current_name = MODES[current_mode]["name"]
            await message.answer(
                f"🎨 *Режимы ответа*\n\n"
                f"{mode_list}\n\n"
                f"✨ *Текущий режим:* `{current_mode}` — {current_name}\n\n"
                f"📝 *Использование:* `/mode <режим>`\n"
                f"Пример: `/mode detailed`\n\n"
                f"💡 Режим сохраняется и не сбрасывается после перезапуска!",
                parse_mode="Markdown"
            )
            return
        if args not in MODES:
            available = ", ".join(MODES.keys())
            await message.answer(
                f"❌ Режим `{args}` не найден.\n\n"
                f"Доступные режимы: {available}\n"
                f"Используй `/mode` без аргументов для списка.",
                parse_mode="Markdown"
            )
            return
        user_mode[message.from_user.id] = args
        save_user_mode(message.from_user.id, args)
        mode_info = MODES[args]
        await message.answer(
            f"✅ *Режим изменен на {mode_info['emoji']} {mode_info['name']}*\n\n"
            f"📝 *Описание:* {mode_info['system']}\n\n"
            f"💡 Теперь я буду отвечать в этом стиле!",
            parse_mode="Markdown"
        )

    # ==================== КОМАНДА ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ (ОТЛАДОЧНАЯ) ====================

    
    # ==================== АДМИН-КОМАНДЫ ====================
    
    @dp.message(Command("stats"))
    @admin_only
    async def cmd_stats(message: Message) -> None:
        users_count, requests_count = get_stats_from_db()
        uptime_sec = int(time.time() - BOT_STARTED_AT)
        h, rem = divmod(uptime_sec, 3600)
        m, s = divmod(rem, 60)
        active_users = len(processing_user_ids)
        mode_stats = defaultdict(int)
        for uid, mode in user_mode.items():
            mode_stats[mode] += 1
        mode_stats_text = "\n".join([
            f"   {MODES[mode]['emoji']} {mode}: {count} пользователей"
            for mode, count in sorted(mode_stats.items(), key=lambda x: -x[1])
        ]) or "   нет данных"
        await message.answer(
            f"📊 *Статистика PrimeAi* (админ-панель)\n\n"
            f"👥 *Всего пользователей:* {users_count}\n"
            f"🤖 *Запросов к ИИ:* {requests_count}\n"
            f"⚡ *Активных запросов:* {active_users}\n"
            f"⏱️ *Аптайм:* {h}ч {m}м {s}с\n\n"
            f"🎨 *Распределение режимов:*\n{mode_stats_text}\n\n"
            f"🧠 *Модель:* `{OPENROUTER_MODEL}`\n"
            f"📝 *max_tokens:* {OPENROUTER_MAX_TOKENS}\n"
            f"💾 *Хранилище:* SQLite (персистентное)\n"
            f"👑 *Администраторы:* {', '.join(str(uid) for uid in ADMIN_IDS) if ADMIN_IDS else 'не заданы'}",
            parse_mode="Markdown"
        )

    @dp.message(Command("admin"))
    @admin_only
    async def cmd_admin(message: Message) -> None:
        if not ADMIN_IDS:
            await message.answer(
                "🔧 Админка не настроена.\n"
                "Добавь ADMIN_IDS в .env файл.\n"
                "Пример: ADMIN_IDS=123456789"
            )
            return
        users_count, requests_count = get_stats_from_db()
        await message.answer(
            "👑 *Админ-панель PrimeAi*\n\n"
            "Доступные команды:\n"
            "📊 `/stats` — полная статистика бота\n"
            "🗑️ `/admin_clear <user_id>` — очистить историю пользователя\n"
            "🆔 `/whoami` — показать свой Telegram ID\n\n"
            "📁 *Хранилище:* SQLite — данные сохраняются при перезапуске\n"
            f"👥 *Всего пользователей в БД:* {users_count}\n"
            f"🤖 *Всего запросов:* {requests_count}\n"
            f"💾 *База данных:* `{DB_PATH}`",
            parse_mode="Markdown"
        )

    @dp.message(Command("admin_clear"))
    @admin_only
    async def cmd_admin_clear(message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip()
        if not args:
            await message.answer("📝 Использование: `/admin_clear <user_id>`", parse_mode="Markdown")
            return
        if not args.isdigit():
            await message.answer("❌ user_id должен быть числом.")
            return
        uid = int(args)
        history = get_history_from_db(uid, 1)
        if not history:
            await message.answer(f"⚠️ Пользователь с ID `{uid}` не найден в базе данных.", parse_mode="Markdown")
            return
        if uid in chat_history:
            chat_history[uid].clear()
        clear_history_in_db(uid)
        await message.answer(
            f"✅ История диалога для пользователя `{uid}` очищена.",
            parse_mode="Markdown"
        )

    # ==================== ОБРАБОТКА ТЕКСТОВЫХ СООБЩЕНИЙ ====================
    
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
            register_user(user.id)
            increment_ai_requests()
            ack_msg = await message.answer(ACK_TEXT)
            try:
                await bot.send_chat_action(
                    chat_id=message.chat.id,
                    action=ChatAction.TYPING,
                )
                reply = await ask_openrouter(session, user.id, message.text)
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

    # ==================== ПРОВЕРКА ПОДКЛЮЧЕНИЯ ====================
    
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
    logging.info(f"👑 Администраторы: {', '.join(str(uid) for uid in ADMIN_IDS) if ADMIN_IDS else 'не заданы'}")
    
    try:
        await dp.start_polling(bot)
    finally:
        await session.close()
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
