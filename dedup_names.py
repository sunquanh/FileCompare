"""同名相似：去掉「副本 / Copy (2)」等复制标记后按文件名归组。

模糊匹配用倒排索引（词 → 文件）先筛候选，再算相似度，
避免原来的「按长度分桶 + 固定窗口 80」漏掉相邻桶里的候选。
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from dedup_models import FileItem

_LOG = logging.getLogger("dedup.names")

_COPY_MARK = re.compile(
    r"(?i)(?:[\s_\-—–.]+)?(?:"
    r"[\(\[（【]\s*(?:副本|拷贝|复制|copy|backup|bak)?\s*\d{0,3}\s*[\)\]）】]"
    r"|(?:副本|拷贝|复制|copy|backup|bak)(?:[\s_\-—–.]*\d{1,3})?"
    r")[\s_\-—–.]*$")

SPLIT_RE = re.compile(r"[\s_\-.()\[\]{}·,，;；+&@#!！~`'\"、/\\]+")
COMMON_WORD_DF = 50


def name_key(stem: str) -> str:
    """去掉 Windows 复制文件留下的「副本 / - Copy (2)」等标记。"""
    s = (stem or "").strip().lower()
    for _ in range(3):
        new = _COPY_MARK.sub("", s)
        if new == s:
            break
        s = new
    s = re.sub(r"[\s_\-—–.]+$", "", s)
    return s


def name_tokens(key: str) -> list[str]:
    return [t for t in SPLIT_RE.split(key or "") if len(t) >= 2]


def find_name_groups(files, max_group=24, fuzzy=False, ratio=0.88):
    """返回 (groups, oversized, fuzzy_groups)。

    groups 中每组是 [FileItem]，按可释放空间降序。
    """
    buckets: dict[str, list[FileItem]] = {}
    for f in files:
        key = name_key(f.path.stem)
        if not key:
            continue
        buckets.setdefault(key, []).append(f)

    out: list[list[FileItem]] = []
    oversized = 0
    singles: list[tuple[str, FileItem]] = []
    for key, items in buckets.items():
        if len(items) < 2:
            if fuzzy and len(key) >= 5:
                singles.append((key, items[0]))
            continue
        if len(items) > max_group:
            oversized += 1
            continue
        out.append(sorted(items, key=lambda x: str(x.path)))

    fuzzy_groups = 0
    if fuzzy and len(singles) >= 2:
        index: dict[str, list[int]] = {}
        for idx, (key, _item) in enumerate(singles):
            for token in set(name_tokens(key)):
                index.setdefault(token, []).append(idx)
        counter: dict[tuple, int] = {}
        for token, ids in index.items():
            if len(ids) > COMMON_WORD_DF or len(ids) < 2:
                continue
            ids = sorted(ids)
            for pos, i in enumerate(ids):
                for j in ids[pos + 1:]:
                    counter[(i, j)] = counter.get((i, j), 0) + 1

        parent = list(range(len(singles)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (i, j), shared in counter.items():
            ki, fi = singles[i]
            kj, fj = singles[j]
            if fi.path.suffix.lower() != fj.path.suffix.lower():
                continue
            if abs(len(ki) - len(kj)) > 4:
                continue
            small, big = sorted((fi.size, fj.size))
            if small <= 0 or big / small > 1.35:
                continue
            # 纯编号不同（IMG_1001 / IMG_1002）不算相似
            if re.sub(r"\d+", "", ki) == re.sub(r"\d+", "", kj):
                continue
            dice = 2.0 * shared / (len(name_tokens(ki)) + len(name_tokens(kj))
                                   or 1)
            if dice < ratio * 0.8:
                continue
            try:
                score = SequenceMatcher(None, ki, kj).ratio()
            except Exception as exc:
                _LOG.debug("相似度计算失败 %r / %r: %s", ki, kj, exc)
                continue
            if score >= ratio:
                union(i, j)

        groups: dict[int, list[FileItem]] = {}
        for idx in range(len(singles)):
            groups.setdefault(find(idx), []).append(singles[idx][1])
        for items in groups.values():
            if len(items) >= 2:
                fuzzy_groups += 1
                out.append(sorted(items, key=lambda x: str(x.path)))

    out.sort(key=lambda v: -sum(i.size for i in v))
    return out, oversized, fuzzy_groups
