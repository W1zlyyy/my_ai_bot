import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Set

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from dotenv import load_dotenv


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
FALLBACK_OPENROUTER_MODEL = "z-ai/glm-4.5-air"
OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", FALLBACK_OPENROUTER_MODEL).strip().strip("\"'")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()
# Сообщение сразу после текста пользователя (до ответа модели)
ACK_TEXT = os.getenv(
    "BOT_ACK_TEXT",
    "Запрос принят, обрабатываю…",
).strip() or "Запрос принят, обрабатываю…"
# Если пользователь пишет, пока идёт ответ на прошлое сообщение
BUSY_TEXT = os.getenv(
    "BOT_BUSY_TEXT",
    "Подожди, сейчас обрабатываю твоё прошлое сообщение.",
).strip() or "Подожди, сейчас обрабатываю твоё прошлое сообщение."

MAX_HISTORY_MESSAGES = 20

BOT_STARTED_AT = time.time()

# user_id -> deque of messages in OpenAI/OpenRouter format
chat_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)

# Статистика (в памяти; после перезапуска обнуляется)
stats_user_ids: Set[int] = set()
stats_ai_requests: int = 0

# Пользователи, у которых сейчас выполняется запрос к ИИ (один активный запрос на user_id)
processing_user_ids: Set[int] = set()


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


def register_user(user_id: int) -> None:
    stats_user_ids.add(user_id)


def validate_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")


async def ask_openrouter(
    session: aiohttp.ClientSession, user_id: int, user_text: str
) -> str:
    history = chat_history[user_id]
    history.append({"role": "user", "content": user_text})

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful Telegram assistant. "
                "Reply briefly and clearly in the same language as the user."
            ),
        },
        *list(history),
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
                    error_text = await retry_resp.text()
                    history.pop()
                    raise RuntimeError(f"OpenRouter error {retry_resp.status}: {error_text}")
                data = await retry_resp.json()
        elif resp.status != 200:
            error_text = await resp.text()
            history.pop()
            if resp.status == 402:
                raise RuntimeError(
                    "Недостаточно кредитов OpenRouter или слишком большой лимит ответа. "
                    "Пополните баланс либо уменьшите OPENROUTER_MAX_TOKENS в .env "
                    "(например, до 256-512)."
                )
            raise RuntimeError(f"OpenRouter error {resp.status}: {error_text}")
        else:
            data = await resp.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Извини, не смог сгенерировать ответ.")
        )

    history.append({"role": "assistant", "content": answer})
    return answer


async def main() -> None:
    validate_env()

    logging.basicConfig(level=logging.INFO)
    tg_session = AiohttpSession(proxy=TELEGRAM_PROXY_URL or None)
    bot = Bot(token=BOT_TOKEN, session=tg_session)
    dp = Dispatcher()

    session = aiohttp.ClientSession()

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        if message.from_user:
            register_user(message.from_user.id)
        await message.answer(
            "Добро пожаловать в PrimeAi бота.\n" 
            "Задай мне любой вопрос и получи мгновенный ответ.\n"
            "Команда /clear очищает историю."
        )

    @dp.message(Command("whoami"))
    async def cmd_whoami(message: Message) -> None:
        uid = message.from_user.id if message.from_user else None
        await message.answer(
            f"Ваш Telegram ID: {uid}\n"
            "Его можно указать в ADMIN_IDS в .env для доступа к админ-командам."
        )

    @dp.message(Command("admin"))
    async def cmd_admin(message: Message) -> None:
        if not message.from_user:
            return
        if not ADMIN_IDS:
            await message.answer(
                "Админка не настроена: в .env пустой или отсутствует ADMIN_IDS "
                "(через запятую, например: ADMIN_IDS=123456789). "
                "Узнай свой ID командой /whoami."
            )
            return
        if not is_admin(message.from_user.id):
            await message.answer("Нет доступа.")
            return
        await message.answer(
            "Админ-команды:\n"
            "/stats — пользователи, запросы к ИИ, аптайм\n"
            "/admin_clear <user_id> — очистить историю диалога пользователю\n"
            "/whoami — показать свой Telegram ID (для всех)"
        )

    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        if not message.from_user:
            return
        if not ADMIN_IDS:
            await message.answer("ADMIN_IDS не задан в .env — админка отключена.")
            return
        if not is_admin(message.from_user.id):
            await message.answer("Нет доступа.")
            return
        uptime_sec = int(time.time() - BOT_STARTED_AT)
        h, rem = divmod(uptime_sec, 3600)
        m, s = divmod(rem, 60)
        await message.answer(
            "Статистика бота:\n"
            f"• Уникальных пользователей: {len(stats_user_ids)}\n"
            f"• Запросов к ИИ (сообщений): {stats_ai_requests}\n"
            f"• Аптайм: {h}ч {m}м {s}с\n"
            f"• Модель: {OPENROUTER_MODEL}\n"
            f"• max_tokens: {OPENROUTER_MAX_TOKENS}"
        )

    @dp.message(Command("admin_clear"))
    async def cmd_admin_clear(message: Message, command: CommandObject) -> None:
        if not message.from_user:
            return
        if not ADMIN_IDS:
            await message.answer("ADMIN_IDS не задан в .env.")
            return
        if not is_admin(message.from_user.id):
            await message.answer("Нет доступа.")
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
        await message.answer(f"История диалога для user_id={uid} очищена.")

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        chat_history[message.from_user.id].clear()
        await message.answer("История диалога очищена.")

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
            global stats_ai_requests
            stats_ai_requests += 1

            ack_msg = await message.answer(ACK_TEXT)
            try:
                await bot.send_chat_action(
                    chat_id=message.chat.id,
                    action=ChatAction.TYPING,
                )
                reply = await ask_openrouter(session, user.id, message.text)
                await message.answer(reply)
            except Exception as exc:
                logging.exception("Failed to handle message")
                await message.answer(f"Ошибка при обращении к OpenRouter: {exc}")
            finally:
                try:
                    await bot.delete_message(
                        chat_id=ack_msg.chat.id,
                        message_id=ack_msg.message_id,
                    )
                except Exception:
                    # Нет прав в группе, сообщение уже удалено и т.д.
                    logging.debug("Could not delete ack message", exc_info=True)
        finally:
            processing_user_ids.discard(user.id)

    # Verify Telegram connectivity before long polling to surface network issues early.
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

    try:
        await dp.start_polling(bot)
    finally:
        await session.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())