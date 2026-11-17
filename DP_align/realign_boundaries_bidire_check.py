#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realign_boundaries.py

Minimal local boundary resegmentation for coarse Vecalign JSONL lines.

Assumptions:
- Coarse file is JSONL with fields such as:
    chapter_id
    parent_en_para_ids
    parent_zh_para_ids
    en_text
    zh_text
    en_tokens
    zh_tokens
    similarity
    vecalign_score

- Paragraph files are JSONL with fields such as:
    chapter_id
    para_id
    embedding
    token
    text

- Use separate paragraph files for Chinese and English:
    --zh-paras
    --en-paras

Output:
- Realigned coarse JSONL.
- Optional JSONL log.
"""

import argparse
import json
import math
import sys


def eprint(*args):
    print(*args, file=sys.stderr)


def get_field(obj, names, default=None):
    """
    Get first non-None field from obj among candidate names.
    """
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def normalize_embedding(emb):
    """
    Normalize embedding field into list[float].
    Supports:
    - list/tuple
    - JSON string: "[0.1, 0.2]"
    - comma-separated string: "0.1,0.2"
    - space-separated string: "0.1 0.2"
    """
    if emb is None:
        return None

    if isinstance(emb, (list, tuple)):
        try:
            return [float(x) for x in emb]
        except Exception:
            return None

    if isinstance(emb, str):
        s = emb.strip()
        if not s:
            return None

        # JSON list string
        if s.startswith("["):
            try:
                val = json.loads(s)
                if isinstance(val, list):
                    return [float(x) for x in val]
            except Exception:
                pass

        # comma or space separated string
        for sep in [",", " "]:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                try:
                    return [float(p) for p in parts]
                except Exception:
                    continue

        # single float string
        try:
            return [float(s)]
        except Exception:
            return None

    return None


def token_count_from_field(token_field, text):
    """
    Convert token field into integer token count.

    Supports:
    - int / float token count
    - list of tokens
    - string token count
    - string tokens separated by whitespace
    - None => use text.split()
    """
    if token_field is None:
        return len(text.split()) if text else 0

    if isinstance(token_field, bool):
        return 0

    if isinstance(token_field, (int, float)):
        return max(0, int(token_field))

    if isinstance(token_field, list):
        return len(token_field)

    if isinstance(token_field, str):
        s = token_field.strip()
        if not s:
            return len(text.split()) if text else 0
        try:
            return max(0, int(float(s)))
        except Exception:
            # Maybe it is a whitespace-separated token list.
            return len(s.split())

    return 0

def norm_chapter(ch):
    """
    Normalize chapter_id / chapter value into string key.
    """
    if ch is None:
        return ""
    return str(ch).strip()

def load_paragraphs_by_chapter(path):
    """
    Load paragraph JSONL file grouped by chapter.

    Expected fields:
        chapter_id, para_id, embedding, token, text

    Returns:
        by_chapter:
            chapter_key -> para_id -> {
                "chapter_id": original chapter value,
                "embedding": list[float] or None,
                "token_count": int,
                "text": str
            }

        avg_token_count:
            global average token count
    """
    by_chapter = {}
    token_counts = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as exc:
                eprint(f"[WARN] {path}:{line_no}: cannot parse JSON: {exc}")
                continue

            pid = get_field(obj, ["para_id", "id", "paragraph_id"])
            if pid is None:
                eprint(f"[WARN] {path}:{line_no}: missing para_id; skipped")
                continue

            pid = str(pid)

            chapter_raw = get_field(obj, ["chapter_id", "chapter"])
            chapter_key = norm_chapter(chapter_raw)

            embedding = normalize_embedding(
                get_field(obj, ["embedding", "emb", "vector"])
            )

            text = get_field(obj, ["text", "para_text"], "")
            if text is None:
                text = ""
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)
            if not isinstance(text, str):
                text = str(text)

            token_field = get_field(
                obj,
                ["token", "tokens", "token_count", "tokens_count"]
            )
            token_count = token_count_from_field(token_field, text)

            paras = by_chapter.setdefault(chapter_key, {})

            if pid in paras:
                eprint(
                    f"[WARN] {path}:{line_no}: duplicate para_id {pid} "
                    f"inside chapter {chapter_raw}; overwriting"
                )

            paras[pid] = {
                "chapter_id": chapter_raw,
                "embedding": embedding,
                "token_count": token_count,
                "text": text,
            }

            if token_count > 0:
                token_counts.append(token_count)

    avg_token_count = (
        sum(token_counts) / len(token_counts)
        if token_counts
        else 1.0
    )

    if avg_token_count <= 0:
        avg_token_count = 1.0

    return by_chapter, avg_token_count

def parse_id_list(value):
    """
    Parse paragraph id list.

    Supports:
    - list
    - tuple
    - JSON list string
    - comma-separated string
    - single string id
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass

        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]

        return [s]

    return [str(value)]


def load_coarse(path):
    """
    Load coarse Vecalign JSONL.
    Normalize parent_zh_para_ids and parent_en_para_ids into lists.
    """
    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as exc:
                eprint(f"[WARN] {path}:{line_no}: cannot parse JSON: {exc}")
                continue

            obj["parent_zh_para_ids"] = parse_id_list(
                get_field(
                    obj,
                    [
                        "parent_zh_para_ids",
                        "zh_para_ids",
                        "zh_ids",
                        "zh_paragraph_ids",
                    ],
                    [],
                )
            )

            obj["parent_en_para_ids"] = parse_id_list(
                get_field(
                    obj,
                    [
                        "parent_en_para_ids",
                        "en_para_ids",
                        "en_ids",
                        "en_paragraph_ids",
                    ],
                    [],
                )
            )

            if "chapter_id" not in obj or obj["chapter_id"] is None:
                obj["chapter_id"] = get_field(obj, ["chapter_id", "chapter"])

            lines.append(obj)

    return lines


def cosine(a, b):
    """
    Pure Python cosine similarity.
    """
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def raw_match_score(zh_para, en_para, avg_zh_tokens, avg_en_tokens, args):
    """
    Raw paragraph match score in [0, 1]-like scale.

    score = cosine * length_factor

    length_factor = exp(-lambda_len * abs(log(len_zh / len_en)))
    where lengths are normalized by average token counts.
    """
    if zh_para is None or en_para is None:
        return 0.0

    emb_zh = zh_para.get("embedding")
    emb_en = en_para.get("embedding")

    if emb_zh is None or emb_en is None:
        return 0.0

    c = cosine(emb_zh, emb_en)

    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0

    len_zh = max(1, int(zh_para.get("token_count", 1) or 1))
    len_en = max(1, int(en_para.get("token_count", 1) or 1))

    ratio = math.log(
        ((len_zh / avg_zh_tokens) + 1e-9)
        /
        ((len_en / avg_en_tokens) + 1e-9)
    )

    length_factor = math.exp(-args.lambda_len * abs(ratio))

    return c * length_factor


def get_adjusted_match(
    chapter_key,
    zh_id,
    en_id,
    zh_paras,
    en_paras,
    avg_zh_tokens,
    avg_en_tokens,
    args,
    cache,
):
    """
    Adjusted match score used by DP.

    cache key includes chapter_key to avoid collision when para_id
    is only unique inside a chapter.
    """
    key = (chapter_key, zh_id, en_id)

    if key in cache:
        return cache[key]

    zh_para = zh_paras.get(zh_id)
    en_para = en_paras.get(en_id)

    if (
        zh_para is None
        or en_para is None
        or zh_para.get("embedding") is None
        or en_para.get("embedding") is None
    ):
        val = -1e9
    else:
        raw = raw_match_score(
            zh_para,
            en_para,
            avg_zh_tokens,
            avg_en_tokens,
            args,
        )
        val = raw - args.match_threshold

    cache[key] = val
    return val

def build_match_matrix(
    chapter_key,
    zh_ids,
    en_ids,
    zh_paras,
    en_paras,
    avg_zh_tokens,
    avg_en_tokens,
    args,
    cache,
):
    """
    Build match matrix for all zh/en paragraph pairs in local boundary window.

    chapter_key is included in cache keys.
    """
    matrix = {}

    unique_zh = list(dict.fromkeys(zh_ids))
    unique_en = list(dict.fromkeys(en_ids))

    for zh_id in unique_zh:
        for en_id in unique_en:
            matrix[(zh_id, en_id)] = get_adjusted_match(
                chapter_key,
                zh_id,
                en_id,
                zh_paras,
                en_paras,
                avg_zh_tokens,
                avg_en_tokens,
                args,
                cache,
            )

    return matrix


def apply_shift_to_sides(left_seq, right_seq, shift):
    """
    Apply boundary shift to one language side.

    shift > 0:
        Move last `shift` paragraphs from left line to right line head.

    shift < 0:
        Move first `-shift` paragraphs from right line to left line tail.

    Returns:
        (new_left_seq, new_right_seq)
    """
    left_seq = list(left_seq)
    right_seq = list(right_seq)

    if shift == 0:
        return left_seq, right_seq

    if shift > 0:
        if shift > len(left_seq):
            return None
        return left_seq[:-shift], left_seq[-shift:] + right_seq

    m = -shift
    if m > len(right_seq):
        return None

    return left_seq + right_seq[:m], right_seq[m:]


def get_moved_ids_detailed(
    left_zh,
    right_zh,
    left_en,
    right_en,
    zh_shift,
    en_shift,
):
    moved_zh = []
    moved_en = []

    if zh_shift > 0:
        moved_zh.extend(left_zh[-zh_shift:])
    elif zh_shift < 0:
        m = -zh_shift
        moved_zh.extend(right_zh[:m])

    if en_shift > 0:
        moved_en.extend(left_en[-en_shift:])
    elif en_shift < 0:
        m = -en_shift
        moved_en.extend(right_en[:m])

    return moved_zh, moved_en


def get_moved_ids(
    left_zh,
    right_zh,
    left_en,
    right_en,
    zh_shift,
    en_shift,
):
    moved_zh, moved_en = get_moved_ids_detailed(
        left_zh,
        right_zh,
        left_en,
        right_en,
        zh_shift,
        en_shift,
    )
    return moved_zh + moved_en


def is_valid_shift(
    left_zh,
    right_zh,
    left_en,
    right_en,
    zh_shift,
    en_shift,
    locked_ids,
    args,
):
    """
    Validate a candidate boundary shift.
    """
    if zh_shift > len(left_zh) or -zh_shift > len(right_zh):
        return False

    if en_shift > len(left_en) or -en_shift > len(right_en):
        return False

    zh_res = apply_shift_to_sides(left_zh, right_zh, zh_shift)
    if zh_res is None:
        return False

    en_res = apply_shift_to_sides(left_en, right_en, en_shift)
    if en_res is None:
        return False

    new_left_zh, new_right_zh = zh_res
    new_left_en, new_right_en = en_res

    # Each resulting line must contain at least something.
    if len(new_left_zh) + len(new_left_en) == 0:
        return False

    if len(new_right_zh) + len(new_right_en) == 0:
        return False

    # Optional: require both languages in every line.
    if args.require_both_languages:
        if (
            len(new_left_zh) == 0
            or len(new_left_en) == 0
            or len(new_right_zh) == 0
            or len(new_right_en) == 0
        ):
            return False

    # Optional maximum line length per language side.
    if args.max_paragraphs_per_line > 0:
        if (
            len(new_left_zh) > args.max_paragraphs_per_line
            or len(new_left_en) > args.max_paragraphs_per_line
            or len(new_right_zh) > args.max_paragraphs_per_line
            or len(new_right_en) > args.max_paragraphs_per_line
        ):
            return False

    # Do not move locked paragraphs.
    moved = get_moved_ids(
        left_zh,
        right_zh,
        left_en,
        right_en,
        zh_shift,
        en_shift,
    )

    if locked_ids.intersection(moved):
        return False

    return True


def dp_align(zh_ids, en_ids, match_matrix, gap_penalty):
    """
    Monotonic alignment DP for one line.

    Allows:
    - match
    - skip zh paragraph
    - skip en paragraph

    This is intentionally simple and minimal.
    """
    m = len(zh_ids)
    n = len(en_ids)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for a in range(1, m + 1):
        dp[a][0] = dp[a - 1][0] - gap_penalty

    for b in range(1, n + 1):
        dp[0][b] = dp[0][b - 1] - gap_penalty

    for a in range(1, m + 1):
        zh_id = zh_ids[a - 1]
        for b in range(1, n + 1):
            en_id = en_ids[b - 1]

            match_score = match_matrix.get((zh_id, en_id), -1e9)

            dp[a][b] = max(
                dp[a - 1][b - 1] + match_score,
                dp[a - 1][b] - gap_penalty,
                dp[a][b - 1] - gap_penalty,
            )

    return dp[m][n]


def normalized_line_score(zh_ids, en_ids, match_matrix, gap_penalty):
    score = dp_align(zh_ids, en_ids, match_matrix, gap_penalty)
    total = len(zh_ids) + len(en_ids)
    return score / max(1, total)


def compute_balance_penalty(left_zh, left_en, right_zh, right_en, args):
    """
    Optional penalty for extremely unbalanced zh/en paragraph counts.

    Default is usually 0 because omission/addition can be translation style.
    """
    if args.balance_penalty == 0.0:
        return 0.0

    pen = 0

    pen += max(
        0,
        abs(len(left_zh) - len(left_en)) - args.balance_tolerance
    )

    pen += max(
        0,
        abs(len(right_zh) - len(right_en)) - args.balance_tolerance
    )

    return args.balance_penalty * float(pen)


def get_vecalign_confidence(left_line, right_line, args):
    """
    Optional prior from original vecalign_score.
    vecalign_score is a COST: lower means more trustworthy.
    confidence = 1 - clip(mean_cost / vecalign_scale, 0, 1)
    Low cost  -> confidence near 1 -> heavier move penalty (protect).
    High cost -> confidence near 0 -> base penalty only (allow repair).
    Missing scores -> 0.0 (no extra prior).
    """
    vals = []
    for line in (left_line, right_line):
        v = line.get("vecalign_score")
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return 0.0
    cost = sum(vals) / len(vals)
    if args.vecalign_scale > 0:
        cost = cost / args.vecalign_scale
    if cost < 0.0:
        cost = 0.0
    if cost > 1.0:
        cost = 1.0
    return 1.0 - cost

def compute_bidirectional_bonus(
    new_left_zh, new_left_en,
    new_right_zh, new_right_en,
    match_matrix,
    args,
):
    """
    檢查移動後的邊界附近，匹配是否雙向一致。
    如果移動的段落在目標行中不是雙向最佳匹配，給予懲罰。
    """
    bonus = 0.0

    # 只檢查邊界附近的段落（前2段和後2段）
    boundary_zh = (new_left_zh[-2:] if len(new_left_zh) >= 2 else new_left_zh) + \
                  (new_right_zh[:2] if len(new_right_zh) >= 2 else new_right_zh)
    boundary_en = (new_left_en[-2:] if len(new_left_en) >= 2 else new_left_en) + \
                  (new_right_en[:2] if len(new_right_en) >= 2 else new_right_en)

    for zh_id in boundary_zh:
        # 找這個中文段落的最佳英文匹配
        best_en = None
        best_score = -1e9
        for en_id in boundary_en:
            s = match_matrix.get((zh_id, en_id), -1e9)
            if s > best_score:
                best_score = s
                best_en = en_id

        if best_en is not None and best_score > 0:
            # 反向：那個英文段落是否也選這個中文？
            reverse_best_zh = None
            reverse_best_score = -1e9
            for zh_id2 in boundary_zh:
                s = match_matrix.get((zh_id2, best_en), -1e9)
                if s > reverse_best_score:
                    reverse_best_score = s
                    reverse_best_zh = zh_id2

            if reverse_best_zh != zh_id:
                # 單向匹配，不可靠，懲罰
                bonus -= 0.03
            else:
                # 雙向一致，獎勵
                bonus += 0.01

    return bonus
    
def score_candidate(
    left_zh,
    right_zh,
    left_en,
    right_en,
    zh_shift,
    en_shift,
    match_matrix,
    args,
    vecalign_confidence,
):
    """
    Score one candidate boundary segmentation.

    candidate_score =
        normalized_align_score
      - move_penalty
      - balance_penalty
    """
    if not is_valid_shift(
        left_zh,
        right_zh,
        left_en,
        right_en,
        zh_shift,
        en_shift,
        set(),  # locked IDs are checked outside / before candidate generation
        args,
    ):
        return float("-inf")

    zh_res = apply_shift_to_sides(left_zh, right_zh, zh_shift)
    en_res = apply_shift_to_sides(left_en, right_en, en_shift)

    if zh_res is None or en_res is None:
        return float("-inf")

    new_left_zh, new_right_zh = zh_res
    new_left_en, new_right_en = en_res

    score_left = dp_align(
        new_left_zh,
        new_left_en,
        match_matrix,
        args.gap_penalty,
    )

    score_right = dp_align(
        new_right_zh,
        new_right_en,
        match_matrix,
        args.gap_penalty,
    )

    total_paragraphs = (
        len(new_left_zh)
        + len(new_left_en)
        + len(new_right_zh)
        + len(new_right_en)
    )

    normalized_align_score = (score_left + score_right) / max(1, total_paragraphs)

    # ★ 新增：雙向一致性加成
    bidirectional_bonus = compute_bidirectional_bonus(
        new_left_zh, new_left_en,
        new_right_zh, new_right_en,
        match_matrix,
        args,
    )

    move_penalty = args.move_penalty * (abs(zh_shift) + abs(en_shift))
    if args.vecalign_weight != 0.0:
        move_penalty *= (1.0 + args.vecalign_weight * vecalign_confidence)

    balance_penalty = compute_balance_penalty(
        new_left_zh,
        new_left_en,
        new_right_zh,
        new_right_en,
        args,
    )

    return normalized_align_score + bidirectional_bonus - move_penalty - balance_penalty

def get_line_id(line, idx):
    return line.get("line_id", line.get("id", idx))


def optimize_boundary(
    left_line,
    right_line,
    left_index,
    right_index,
    chapter_key,
    zh_paras,
    en_paras,
    avg_zh_tokens,
    avg_en_tokens,
    args,
    match_cache,
    locked_ids,
):
    """
    Optimize one boundary between left_line and right_line.

    zh_paras / en_paras are chapter-specific paragraph dictionaries:

        para_id -> paragraph_info

    chapter_key is used for cache keys and chapter-scoped lookup.
    """
    left_zh = list(left_line.get("parent_zh_para_ids", []))
    left_en = list(left_line.get("parent_en_para_ids", []))
    right_zh = list(right_line.get("parent_zh_para_ids", []))
    right_en = list(right_line.get("parent_en_para_ids", []))

    left_ch = norm_chapter(left_line.get("chapter_id"))
    right_ch = norm_chapter(right_line.get("chapter_id"))

    if not chapter_key:
        chapter_key = left_ch

    info = {
        "boundary_index": left_index,
        "left_line_id": get_line_id(left_line, left_index),
        "right_line_id": get_line_id(right_line, right_index),
        "chapter_id": left_line.get("chapter_id"),
        "chapter_key": chapter_key,
    }

    # Do not move across chapters by default.
    if not args.allow_cross_chapter:
        if left_ch != right_ch:
            info["skip"] = "different_chapter"
            return None, info, None

    if not zh_paras or not en_paras:
        info["skip"] = "missing_chapter_paragraph_dict"
        return None, info, None

    all_zh = left_zh + right_zh
    all_en = left_en + right_en

    # If one language is completely absent, local bilingual scoring is unstable.
    if not all_zh or not all_en:
        info["skip"] = "empty_language_side"
        return None, info, None

    # Check missing embeddings inside this chapter.
    missing = []

    for pid in all_zh:
        p = zh_paras.get(pid)
        if p is None or p.get("embedding") is None:
            missing.append(pid)

    for pid in all_en:
        p = en_paras.get(pid)
        if p is None or p.get("embedding") is None:
            missing.append(pid)

    if missing and not args.allow_missing_embeddings:
        info["skip"] = "missing_embeddings"
        info["missing_sample"] = missing[:20]
        return None, info, None

    match_matrix = build_match_matrix(
        chapter_key,
        all_zh,
        all_en,
        zh_paras,
        en_paras,
        avg_zh_tokens,
        avg_en_tokens,
        args,
        match_cache,
    )

    vecalign_confidence = get_vecalign_confidence(left_line, right_line, args)

    current_score = score_candidate(
        left_zh,
        right_zh,
        left_en,
        right_en,
        0,
        0,
        match_matrix,
        args,
        vecalign_confidence,
    )

    if current_score == float("-inf"):
        info["skip"] = "invalid_current_segmentation"
        return None, info, None

    best_score = current_score
    best_zh_shift = 0
    best_en_shift = 0

    for zh_shift in range(-args.max_shift, args.max_shift + 1):
        for en_shift in range(-args.max_shift, args.max_shift + 1):
            if zh_shift == 0 and en_shift == 0:
                continue

            if not is_valid_shift(
                left_zh,
                right_zh,
                left_en,
                right_en,
                zh_shift,
                en_shift,
                locked_ids,
                args,
            ):
                continue

            candidate_score = score_candidate(
                left_zh,
                right_zh,
                left_en,
                right_en,
                zh_shift,
                en_shift,
                match_matrix,
                args,
                vecalign_confidence,
            )

            if candidate_score == float("-inf"):
                continue

            move_size = abs(zh_shift) + abs(en_shift)
            best_move_size = abs(best_zh_shift) + abs(best_en_shift)

            if (
                candidate_score > best_score + 1e-12
                or (
                    abs(candidate_score - best_score) <= 1e-12
                    and move_size < best_move_size
                )
            ):
                best_score = candidate_score
                best_zh_shift = zh_shift
                best_en_shift = en_shift

    gain = best_score - current_score

    info["score_before"] = current_score
    info["score_best"] = best_score
    info["gain"] = gain
    info["best_zh_shift"] = best_zh_shift
    info["best_en_shift"] = best_en_shift

    if gain > args.min_gain and (best_zh_shift != 0 or best_en_shift != 0):
        proposal = {
            "zh_shift": best_zh_shift,
            "en_shift": best_en_shift,
            "score_before": current_score,
            "score_after": best_score,
            "gain": gain,
        }
        return proposal, info, match_matrix

    return None, info, None


def build_tokens(para_ids, paras, original_tokens):
    """
    Rebuild tokens field.

    If original tokens field is int, output summed token count.
    If original tokens field is list, output per-paragraph token counts.
    If absent, output per-paragraph token counts.
    """
    counts = []

    for pid in para_ids:
        p = paras.get(pid)
        if p is None:
            counts.append(0)
        else:
            counts.append(int(p.get("token_count", 0) or 0))

    if isinstance(original_tokens, bool):
        return counts

    if isinstance(original_tokens, int):
        return sum(counts)

    if isinstance(original_tokens, float):
        return int(sum(counts))

    if isinstance(original_tokens, list):
        return counts

    if isinstance(original_tokens, str):
        s = original_tokens.strip()
        try:
            float(s)
            return int(sum(counts))
        except Exception:
            return counts

    return counts


def update_line(
    original_line,
    zh_ids,
    en_ids,
    zh_paras,
    en_paras,
    realigned,
    align_score=None,
):
    """
    Build updated JSONL line object.

    Important:
    - chapter_id / chapter must come from original coarse line.
    - paragraph lookup dictionaries zh_paras / en_paras must already be
      chapter-specific.
    """
    new_line = dict(original_line)

    # Explicitly preserve chapter metadata from coarse JSONL.
    if "chapter_id" in original_line:
        new_line["chapter_id"] = original_line["chapter_id"]

    if "chapter" in original_line:
        new_line["chapter"] = original_line["chapter"]

    new_line["parent_zh_para_ids"] = list(zh_ids)
    new_line["parent_en_para_ids"] = list(en_ids)

    zh_texts = []
    for pid in zh_ids:
        p = zh_paras.get(pid)
        if p is None:
            zh_texts.append("")
        else:
            zh_texts.append(p.get("text", ""))

    en_texts = []
    for pid in en_ids:
        p = en_paras.get(pid)
        if p is None:
            en_texts.append("")
        else:
            en_texts.append(p.get("text", ""))

    new_line["zh_text"] = "\n".join(zh_texts)
    new_line["en_text"] = "\n".join(en_texts)

    new_line["zh_tokens"] = build_tokens(
        zh_ids,
        zh_paras,
        original_line.get("zh_tokens")
    )

    new_line["en_tokens"] = build_tokens(
        en_ids,
        en_paras,
        original_line.get("en_tokens")
    )

    new_line["realigned"] = bool(realigned)

    if align_score is not None and math.isfinite(align_score):
        new_line["local_align_score"] = round(float(align_score), 6)

    return new_line

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal local boundary resegmentation for coarse Vecalign JSONL."
        )
    )

    parser.add_argument(
        "--coarse",
        required=True,
        help="Input coarse Vecalign JSONL file.",
    )

    parser.add_argument(
        "--zh-paras",
        required=True,
        help="Chinese paragraph JSONL file with para_id, embedding, token, text.",
    )

    parser.add_argument(
        "--en-paras",
        required=True,
        help="English paragraph JSONL file with para_id, embedding, token, text.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output realigned coarse JSONL file.",
    )

    parser.add_argument(
        "--log",
        default=None,
        help="Optional JSONL log file.",
    )

    parser.add_argument(
        "--max-shift",
        type=int,
        default=1,
        help="Maximum number of paragraphs to move per language side. Default: 1.",
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.20,
        help=(
            "Raw match score threshold. "
            "DP uses raw_score - match_threshold. Default: 0.20."
        ),
    )

    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.05,
        help=(
            "Penalty for skipping one paragraph. "
            "Small value tolerates omission/addition. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--move-penalty",
        type=float,
        default=0.03,
        help="Penalty per moved paragraph. Default: 0.03.",
    )

    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="Minimum normalized score gain to accept a move. Default: 0.01.",
    )

    parser.add_argument(
        "--lambda-len",
        type=float,
        default=0.20,
        help="Strength of length-ratio penalty. Default: 0.20.",
    )

    parser.add_argument(
        "--balance-penalty",
        type=float,
        default=0.0,
        help=(
            "Optional penalty for zh/en paragraph-count imbalance. "
            "Default: 0.0 because omission/addition may be translation style."
        ),
    )

    parser.add_argument(
        "--balance-tolerance",
        type=int,
        default=2,
        help="Tolerance for zh/en paragraph-count imbalance. Default: 2.",
    )

    parser.add_argument(
            "--vecalign-weight",
            type=float,
            default=0.0,
            help=(
                'If > 0, use vecalign cost as prior. '
                'Lower original cost (more trustworthy) increases move penalty. Default: 0.0.'
            ),
        )
    
    parser.add_argument(
        "--vecalign-scale",
        type=float,
        default=1.0,
        help=(
            "Cost scale: costs >= this value are treated as fully untrustworthy (confidence 0). "
            "Calibrate to a high percentile (e.g., p90) of observed vecalign_score. Default: 1.0."
        ),
    )

    parser.add_argument(
        "--require-both-languages",
        action="store_true",
        help="If set, every output line must contain both zh and en paragraphs.",
    )

    parser.add_argument(
        "--allow-cross-chapter",
        action="store_true",
        help="If set, allow moving paragraphs across different chapter_id.",
    )

    parser.add_argument(
        "--allow-missing-embeddings",
        action="store_true",
        help=(
            "If set, do not skip boundaries with missing embeddings. "
            "Missing matches receive very low scores."
        ),
    )

    parser.add_argument(
        "--max-paragraphs-per-line",
        type=int,
        default=0,
        help=(
            "Maximum number of paragraphs per language side in a line. "
            "0 means no limit. Default: 0."
        ),
    )

    parser.add_argument(
        "--log-all",
        action="store_true",
        help="If set, log all boundary decisions, including skipped ones.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_shift < 0:
        args.max_shift = abs(args.max_shift)

    if args.vecalign_scale <= 0:
        args.vecalign_scale = 1.0

    eprint(f"[INFO] Loading Chinese paragraphs from {args.zh_paras}")
    zh_paras_by_chapter, avg_zh_tokens = load_paragraphs_by_chapter(args.zh_paras)

    eprint(f"[INFO] Loading English paragraphs from {args.en_paras}")
    en_paras_by_chapter, avg_en_tokens = load_paragraphs_by_chapter(args.en_paras)

    total_zh = sum(len(v) for v in zh_paras_by_chapter.values())
    total_en = sum(len(v) for v in en_paras_by_chapter.values())
    eprint(f"[INFO] Loaded {total_zh} zh paragraphs across {len(zh_paras_by_chapter)} chapters")
    eprint(f"[INFO] Loaded {total_en} en paragraphs across {len(en_paras_by_chapter)} chapters")
    eprint(f"[INFO] avg zh token count = {avg_zh_tokens:.3f}")
    eprint(f"[INFO] avg en token count = {avg_en_tokens:.3f}")

    eprint(f"[INFO] Loading coarse lines from {args.coarse}")
    lines = load_coarse(args.coarse)

    original_zh_ids = [
        pid
        for line in lines
        for pid in line.get("parent_zh_para_ids", [])
    ]

    original_en_ids = [
        pid
        for line in lines
        for pid in line.get("parent_en_para_ids", [])
    ]

    original_chapters = [
        line.get("chapter_id")
        for line in lines
    ]
    eprint(f"[INFO] Loaded {len(lines)} coarse lines")

    # For final consistency validation.
    original_zh_ids = [
        pid
        for line in lines
        for pid in line.get("parent_zh_para_ids", [])
    ]

    original_en_ids = [
        pid
        for line in lines
        for pid in line.get("parent_en_para_ids", [])
    ]

    match_cache = {}
    locked_by_chapter = {}
    logs = []

    accepted_count = 0
    skipped_count = 0

    for i in range(len(lines) - 1):
        left_line = lines[i]
        right_line = lines[i + 1]

        left_ch = norm_chapter(left_line.get("chapter_id"))
        right_ch = norm_chapter(right_line.get("chapter_id"))

        # Default: do not process across chapters.
        if not args.allow_cross_chapter and left_ch != right_ch:
            skipped_count += 1
            if args.log_all:
                logs.append({
                    "boundary_index": i,
                    "accepted": False,
                    "skip": "different_chapter",
                    "left_line_id": get_line_id(left_line, i),
                    "right_line_id": get_line_id(right_line, i + 1),
                    "left_chapter": left_line.get("chapter_id"),
                    "right_chapter": right_line.get("chapter_id"),
                })
            continue

        chapter_key = left_ch

        zh_paras = zh_paras_by_chapter.get(chapter_key, {})
        en_paras = en_paras_by_chapter.get(chapter_key, {})

        if not zh_paras or not en_paras:
            skipped_count += 1
            if args.log_all:
                logs.append({
                    "boundary_index": i,
                    "accepted": False,
                    "skip": "missing_chapter_paragraph_dict",
                    "chapter_key": chapter_key,
                    "left_line_id": get_line_id(left_line, i),
                    "right_line_id": get_line_id(right_line, i + 1),
                })
            continue

        # locked IDs are also chapter-specific.
        locked_ids = locked_by_chapter.setdefault(chapter_key, set())

        proposal, info, match_matrix = optimize_boundary(
            left_line,
            right_line,
            i,
            i + 1,
            chapter_key,
            zh_paras,
            en_paras,
            avg_zh_tokens,
            avg_en_tokens,
            args,
            match_cache,
            locked_ids,
        )

        if proposal is None:
            skipped_count += 1
            if args.log_all:
                info["accepted"] = False
                logs.append(info)
            continue

        # Re-check moved IDs and current line state.
        left_zh = list(left_line.get("parent_zh_para_ids", []))
        left_en = list(left_line.get("parent_en_para_ids", []))
        right_zh = list(right_line.get("parent_zh_para_ids", []))
        right_en = list(right_line.get("parent_en_para_ids", []))

        zh_shift = proposal["zh_shift"]
        en_shift = proposal["en_shift"]

        moved_zh, moved_en = get_moved_ids_detailed(
            left_zh,
            right_zh,
            left_en,
            right_en,
            zh_shift,
            en_shift,
        )

        moved_all = moved_zh + moved_en

        if locked_ids.intersection(moved_all):
            skipped_count += 1
            if args.log_all:
                info["accepted"] = False
                info["skip"] = "locked_paragraph_conflict"
                logs.append(info)
            continue

        zh_res = apply_shift_to_sides(left_zh, right_zh, zh_shift)
        en_res = apply_shift_to_sides(left_en, right_en, en_shift)

        if zh_res is None or en_res is None:
            skipped_count += 1
            if args.log_all:
                info["accepted"] = False
                info["skip"] = "invalid_shift_at_apply_time"
                logs.append(info)
            continue

        new_left_zh, new_right_zh = zh_res
        new_left_en, new_right_en = en_res

        left_align_score = normalized_line_score(
            new_left_zh,
            new_left_en,
            match_matrix,
            args.gap_penalty,
        )

        right_align_score = normalized_line_score(
            new_right_zh,
            new_right_en,
            match_matrix,
            args.gap_penalty,
        )

        lines[i] = update_line(
            left_line,
            new_left_zh,
            new_left_en,
            zh_paras,
            en_paras,
            realigned=True,
            align_score=left_align_score,
        )

        lines[i + 1] = update_line(
            right_line,
            new_right_zh,
            new_right_en,
            zh_paras,
            en_paras,
            realigned=True,
            align_score=right_align_score,
        )

        locked_ids.update(moved_all)
        accepted_count += 1

        log_entry = {
            "boundary_index": i,
            "accepted": True,
            "left_line_id": get_line_id(left_line, i),
            "right_line_id": get_line_id(right_line, i + 1),
            "chapter_id": left_line.get("chapter_id"),
            "chapter_key": chapter_key,
            "zh_shift": zh_shift,
            "en_shift": en_shift,
            "score_before": proposal["score_before"],
            "score_after": proposal["score_after"],
            "gain": proposal["gain"],
            "moved_zh_ids": moved_zh,
            "moved_en_ids": moved_en,
        }
        logs.append(log_entry)

    # Add realigned=False to untouched lines.
    for line in lines:
        if "realigned" not in line:
            line["realigned"] = False

    # Final consistency validation.
    new_zh_ids = [
        pid
        for line in lines
        for pid in line.get("parent_zh_para_ids", [])
    ]

    new_en_ids = [
        pid
        for line in lines
        for pid in line.get("parent_en_para_ids", [])
    ]
    
    new_chapters = [
    line.get("chapter_id")
    for line in lines
    ]



    if original_zh_ids != new_zh_ids:
        eprint("[ERROR] Chinese paragraph sequence changed after realignment.")
        eprint("[ERROR] This indicates a bug or invalid move.")
        sys.exit(1)

    if original_en_ids != new_en_ids:
        eprint("[ERROR] English paragraph sequence changed after realignment.")
        eprint("[ERROR] This indicates a bug or invalid move.")
        sys.exit(1)
        
    if original_chapters != new_chapters:
        eprint("[ERROR] chapter_id sequence changed after realignment.")
        eprint("[ERROR] This indicates output metadata was overwritten incorrectly.")
        sys.exit(1)

        eprint(f"[INFO] Accepted boundary moves: {accepted_count}")
        eprint(f"[INFO] Skipped / unchanged boundaries: {skipped_count}")

    with open(args.output, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    eprint(f"[INFO] Wrote realigned coarse JSONL to {args.output}")

    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            for entry in logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        eprint(f"[INFO] Wrote log to {args.log}")


if __name__ == "__main__":
    main()