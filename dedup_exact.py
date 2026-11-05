"""完全重复检测：分层哈希（大小 → 首尾 → 采样 → 全文件）。

与 Czkawka / fclones / dupeGuru 的做法对齐：
  * 小文件（≤1 MB）在首尾阶段直接整读，不再进全文件阶段；
  * 大文件多一级「首 4KB + 尾 4KB + 25%/60% 各 1MB + 尾 1MB」采样，
    >8MB 的文件在进入全量哈希前再筛一轮，显著减少磁盘读取；
  * 任务按批提交，取消时只等当前批次结束，不再排队等几十万个任务。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import blake2b
from pathlib import Path

from dedup_models import CACHE_TAGS, FileItem, check_cancel

HEAD_TAIL = 4 * 1024
SAMPLE_SIZE = 1 << 20
SMALL_FULL = 1 << 20
SAMPLE_MIN = 8 << 20
CHUNK = 1500


class ProgressThrottle:
    """按时间节流进度回调，避免几十万个事件把界面队列打爆。"""

    def __init__(self, callback, interval=0.08):
        self.callback = callback
        self.interval = interval
        self.last = 0.0

    def __call__(self, done, total, message=None):
        if self.callback is None:
            return
        import time
        now = time.perf_counter()
        if done >= total or now - self.last >= self.interval:
            self.last = now
            self.callback(done, total, message or "")


def parallel_map(func, items, workers=4, cancel=None, on_progress=None,
                 chunk=CHUNK):
    """分批提交 + 可取消的并行 map。结果顺序与输入一致，异常项为 None。"""
    total = len(items)
    results = [None] * total
    if not total:
        return results
    workers = max(1, int(workers or 1))
    executor = ThreadPoolExecutor(max_workers=workers)
    done = 0
    try:
        for start in range(0, total, chunk):
            check_cancel(cancel)
            batch = items[start:start + chunk]
            futures = {executor.submit(func, item): start + i
                       for i, item in enumerate(batch)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception:
                    results[index] = None
                done += 1
                if on_progress is not None:
                    on_progress(done, total, None)
            check_cancel(cancel)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


# ---------------------------------------------------------------- 哈希
def _quick_hash(path: Path, size: int) -> str:
    h = blake2b(digest_size=16)
    h.update(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        if size <= SMALL_FULL:
            h.update(f.read())
        else:
            h.update(f.read(HEAD_TAIL))
            f.seek(-HEAD_TAIL, os.SEEK_END)
            h.update(f.read(HEAD_TAIL))
    return h.hexdigest()


def _sample_hash(path: Path, size: int) -> str:
    h = blake2b(digest_size=16)
    h.update(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        for fraction in (0.25, 0.60):
            offset = int(size * fraction)
            offset = min(offset, max(0, size - SAMPLE_SIZE))
            f.seek(offset)
            h.update(f.read(SAMPLE_SIZE))
        f.seek(max(0, size - SAMPLE_SIZE))
        h.update(f.read(SAMPLE_SIZE))
    return h.hexdigest()


def _full_hash(path: Path, size: int | None = None) -> str:
    """全文件 BLAKE2b-256。size 参数仅为统一调用签名。"""
    h = blake2b(digest_size=32)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 阶段
def _stage_groups(items, func, tag, workers, cancel, on_progress, cache,
                  label):
    """对一批文件计算指纹并按 (size, 指纹) 分组，只保留 ≥2 的组。"""
    def work(item):
        if cache is not None:
            hit = cache.get(item, tag)
            if hit:
                return hit
        digest = func(item.path, item.size)
        if cache is not None:
            cache.put(item, tag, digest)
        return digest

    throttle = ProgressThrottle(on_progress)
    digests = parallel_map(work, items, workers=workers, cancel=cancel,
                           on_progress=lambda d, t, m: throttle(d, t, label))
    buckets: dict[tuple, list[FileItem]] = {}
    for item, digest in zip(items, digests):
        if not digest:
            continue
        buckets.setdefault((item.size, digest), []).append(item)
    return [sorted(v, key=lambda x: str(x.path))
            for v in buckets.values() if len(v) > 1]


def find_exact(files, workers=4, cancel=None, on_progress=None, cache=None):
    """返回 [[FileItem, ...], ...]，全部按字节确认一致。"""
    by_size: dict[int, list[FileItem]] = {}
    for f in files:
        by_size.setdefault(f.size, []).append(f)
    candidates = [v for v in by_size.values() if len(v) > 1]
    if not candidates:
        return []

    remaining = _stage_groups([f for g in candidates for f in g], _quick_hash,
                              CACHE_TAGS["quick"], workers, cancel,
                              on_progress, cache, "快速比对")
    if not remaining:
        return []

    # 中段采样：只处理大文件
    big = [f for g in remaining for f in g if f.size > SAMPLE_MIN]
    if big:
        sampled = _stage_groups(big, _sample_hash, CACHE_TAGS["sample"],
                                workers, cancel, on_progress, cache, "采样比对")
        sampled_ids = {id(f) for g in sampled for f in g}
        merged = []
        for group in remaining:
            small_part = [f for f in group if f.size <= SAMPLE_MIN]
            big_part = [f for f in group if f.size > SAMPLE_MIN]
            if len(small_part) > 1:
                merged.append(sorted(small_part, key=lambda x: str(x.path)))
            survivors = [f for f in big_part if id(f) in sampled_ids]
            if len(survivors) > 1:
                merged.append(sorted(survivors, key=lambda x: str(x.path)))
        remaining = [g for g in merged if len(g) > 1]
    if not remaining:
        return []

    groups = _stage_groups([f for g in remaining for f in g], _full_hash,
                           CACHE_TAGS["full"], workers, cancel, on_progress,
                           cache, "完整校验")
    groups.sort(key=lambda g: -g[0].size)
    return groups


def hash_all(files, workers=4, cancel=None, on_progress=None, cache=None):
    """全量哈希（供旧版 CLI 或其他调用方使用）。"""
    return _stage_groups(files, _full_hash, CACHE_TAGS["full"], workers,
                         cancel, on_progress, cache, "完整校验")
