"""SQLite: история диалогов и счётчики (персистентно между перезапусками)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_history (
                user_id INTEGER PRIMARY KEY,
                messages_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bot_users (
                user_id INTEGER PRIMARY KEY,
                first_seen REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_history(user_id: int, db_path: str) -> List[Dict[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT messages_json FROM user_history WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return []
        data = json.loads(row[0])
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict) and "role" in x and "content" in x]
    finally:
        conn.close()


def save_history(user_id: int, messages: List[Dict[str, Any]], db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_history (user_id, messages_json) VALUES (?, ?)",
            (user_id, json.dumps(messages, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_history(user_id: int, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def register_user_db(user_id: int, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO bot_users (user_id) VALUES (?)",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM bot_users")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def incr_ai_requests(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT value FROM kv WHERE key = ?",
            ("total_ai_requests",),
        )
        row = cur.fetchone()
        if row is None:
            new_val = 1
            conn.execute(
                "INSERT INTO kv(key, value) VALUES ('total_ai_requests', ?)",
                (new_val,),
            )
        else:
            new_val = int(row[0]) + 1
            conn.execute(
                "UPDATE kv SET value = ? WHERE key = 'total_ai_requests'",
                (new_val,),
            )
        conn.commit()
        return new_val
    finally:
        conn.close()


def get_total_ai_requests(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT value FROM kv WHERE key = ?",
            ("total_ai_requests",),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()
