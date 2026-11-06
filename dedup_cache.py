"""指纹缓存（SQLite）。

相比原来的单个 JSON：
  * 按 (路径, 类型) 主键增量写入，不再全量读/写几十万条；
  * 大小或修改时间（纳秒）变了就视为失效，不会把过期结果当成重复；
  * 支持多进程同时读写（WAL）；
  * 按最近使用时间清理，不会无限膨胀，也不会出现「到 8 万条就停止写入」。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from dedup_models import app_dir, ignore

SCHEMA = """
CREATE TABLE IF NOT EXISTS fp (
    path TEXT NOT NULL,
    tag  TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    value TEXT NOT NULL,
    seen REAL NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE INDEX IF NOT EXISTS idx_fp_seen ON fp(seen);
"""


class HashCache:
    """指纹缓存。

    get(item, tag) / put(item, tag, value) 的语义与原实现一致，
    只是把存储换成了 SQLite，并按需清理。
    """

    MAX_ROWS = 300_000
    MAX_AGE_DAYS = 120

    def __init__(self, enabled=True, path=None):
        self.enabled = bool(enabled)
        self.path = Path(path) if path else (app_dir() / "dedup_cache.sqlite3")
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn = None
        self._dirty = 0
        if self.enabled:
            self._open()

    # ------------------------------------------------------------ 连接
    def _open(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=15,
                                   check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
        except Exception:
            self._conn = None
            self.enabled = False

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception as exc:
                    ignore(__name__, exc)
                self._conn = None

    # ------------------------------------------------------------ 读写
    @staticmethod
    def _mtime_ns(item) -> int:
        value = getattr(item, "mtime_ns", 0)
        if value:
            return int(value)
        return int(float(getattr(item, "mtime", 0.0)) * 1_000_000_000)

    def get(self, item, tag: str):
        if not self.enabled or self._conn is None:
            return None
        path = str(item.path)
        size = int(item.size)
        mtime = self._mtime_ns(item)
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value FROM fp WHERE path=? AND tag=?", (path, tag)
                ).fetchone()
            if row is None:
                self.misses += 1
                return None
            value = json.loads(row[0])
            if (isinstance(value, dict) and "__size__" in value
                    and (value["__size__"] != size
                         or value.get("__mtime__") != mtime)):
                self.misses += 1
                self._drop(path, tag)
                return None
        except Exception:
            self.misses += 1
            return None
        self.hits += 1
        return value.get("v") if isinstance(value, dict) and "__size__" in value \
            else value

    def put(self, item, tag: str, value):
        if not self.enabled or self._conn is None:
            return
        if tag in ("q3", "s3") and int(item.size) < (1 << 20):
            # 小文件的首尾/采样哈希重算只要读几 KB，不值得占缓存写放大
            return
        payload = json.dumps({"__size__": int(item.size),
                              "__mtime__": self._mtime_ns(item),
                              "v": value}, ensure_ascii=False)
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO fp(path, tag, size, mtime_ns, value, seen)"
                    " VALUES(?,?,?,?,?,?)",
                    (str(item.path), tag, int(item.size), self._mtime_ns(item),
                     payload, time.time()))
                self._dirty += 1
                commit = self._dirty >= 500
                if commit:
                    self._conn.commit()
                    self._dirty = 0
            if commit:
                pass
        except Exception as exc:
            ignore(__name__, exc)

    def _drop(self, path, tag):
        try:
            with self._lock:
                self._conn.execute("DELETE FROM fp WHERE path=? AND tag=?",
                                   (path, tag))
                self._conn.commit()
        except Exception as exc:
            ignore(__name__, exc)

    # ------------------------------------------------------------ 收尾
    def save(self):
        if not self.enabled or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.commit()
                cutoff = time.time() - self.MAX_AGE_DAYS * 86400
                self._conn.execute("DELETE FROM fp WHERE seen < ?", (cutoff,))
                count = self._conn.execute("SELECT COUNT(*) FROM fp").fetchone()[0]
                if count > self.MAX_ROWS:
                    self._conn.execute(
                        "DELETE FROM fp WHERE rowid IN ("
                        " SELECT rowid FROM fp ORDER BY seen ASC LIMIT ?)",
                        (count - self.MAX_ROWS,))
                self._conn.commit()
        except Exception as exc:
            ignore(__name__, exc)

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "size": self.path.stat().st_size if self.path.exists() else 0}


def cache_files(path=None):
    """缓存数据库及 WAL 附属文件。"""
    base = Path(path) if path else (app_dir() / "dedup_cache.sqlite3")
    return [base, Path(str(base) + "-wal"), Path(str(base) + "-shm")]


def cache_info(path=None):
    """返回 (缓存条目数, 占用字节数)，读不到时按 0 计。"""
    base = Path(path) if path else (app_dir() / "dedup_cache.sqlite3")
    count = 0
    if base.exists():
        try:
            conn = sqlite3.connect(f"file:{base}?mode=ro", uri=True, timeout=5)
            count = conn.execute("SELECT COUNT(*) FROM fp").fetchone()[0]
            conn.close()
        except Exception:
            count = 0
    size = sum(f.stat().st_size for f in cache_files(base)
               if f.exists())
    return count, size


def clear_all(path=None):
    """删除缓存数据库文件，返回释放的字节数。扫描进行中不要调用。"""
    freed = 0
    for f in cache_files(path):
        try:
            if f.exists():
                freed += f.stat().st_size
                f.unlink()
        except OSError:
            pass
    return freed
