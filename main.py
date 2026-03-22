import asyncio
import logging
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.filters import Command
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

MAX_HISTORY_MESSAGES = 20

# user_id -> deque of messages in OpenAI/OpenRouter format
chat_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)


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
        await message.answer(
            "Добро пожаловать в PrimeAi бота.\n" 
            "Задай мне любой вопрос и получи мгновенный ответ.\n"
            "Команда /clear очищает историю."
        )

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        chat_history[message.from_user.id].clear()
        await message.answer("История диалога очищена.")

    @dp.message(F.text)
    async def handle_text(message: Message) -> None:
        user = message.from_user
        if user is None or message.text is None:
            return

        try:
            await message.answer(ACK_TEXT)
            await bot.send_chat_action(
                chat_id=message.chat.id,
                action=ChatAction.TYPING,
            )
            reply = await ask_openrouter(session, user.id, message.text)
            await message.answer(reply)
        except Exception as exc:
            logging.exception("Failed to handle message")
            await message.answer(f"Ошибка при обращении к OpenRouter: {exc}")

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