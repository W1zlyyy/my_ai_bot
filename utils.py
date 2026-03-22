"""Утилиты для Telegram."""

from __future__ import annotations


def split_telegram_text(text: str, max_len: int = 4096) -> list[str]:
    """
    Разбивает текст на части не длиннее max_len (лимит Telegram ~4096).
    Старается резать по переводу строки, иначе по символам.
    """
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        else:
            cut = cut + 1
        chunks.append(rest[:cut])
        rest = rest[cut:]
    return chunks
