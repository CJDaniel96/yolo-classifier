"""Durable work journal; publication is recorded before touching source files."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class State:
    def __init__(self, path: Path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, xml_hash TEXT NOT NULL,
            status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, next_try REAL NOT NULL DEFAULT 0,
            error TEXT, payload TEXT NOT NULL DEFAULT '{}'
        )""")
        self.db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")
        self.db.execute("CREATE INDEX IF NOT EXISTS jobs_updated ON jobs(updated DESC)")
        self.db.execute("UPDATE jobs SET status='waiting' WHERE status='processing'")
        self.db.commit()

    @staticmethod
    def row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def discover(self, key: str, source: Path, xml_hash: str, now: float) -> None:
        with self.lock, self.db:
            self.db.execute("""INSERT OR IGNORE INTO jobs
                (id,source,xml_hash,status,created,updated) VALUES (?,?,?,'waiting',?,?)""",
                (key, str(source), xml_hash, now, now))

    def get(self, key: str) -> dict[str, Any]:
        with self.lock:
            return self.row(self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone())

    def find(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
            return None if row is None else self.row(row)

    def jobs(self, *statuses: str) -> list[dict[str, Any]]:
        with self.lock:
            query = "SELECT * FROM jobs"
            if statuses:
                query += " WHERE status IN (" + ",".join("?" for _ in statuses) + ")"
            query += " ORDER BY created,id"
            return [self.row(row) for row in self.db.execute(query, statuses)]

    def update(self, key: str, now: float, **values: Any) -> None:
        if "payload" in values:
            values["payload"] = json.dumps(values["payload"], ensure_ascii=False, allow_nan=False)
        values["updated"] = now
        with self.lock, self.db:
            self.db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
                            (*values.values(), key))

    def counts(self) -> dict[str, int]:
        with self.lock:
            return dict(self.db.execute("SELECT status,count(*) FROM jobs GROUP BY status"))

    def recent(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return [self.row(row) for row in self.db.execute(
                "SELECT * FROM jobs ORDER BY updated DESC,id LIMIT ?", (limit,))]

    def retry_failed(self, now: float) -> None:
        with self.lock, self.db:
            self.db.execute("""UPDATE jobs SET status='waiting', attempts=0,
                next_try=0,error=NULL,created=?,updated=? WHERE status='failed'""", (now, now))

    def close(self) -> None:
        self.db.close()
