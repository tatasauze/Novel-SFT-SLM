#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併四個資料夾（zh、en、tokens、embeddings）為兩個 JSONL 檔：

    zh_paras.jsonl
    en_paras.jsonl

每行欄位：
    chapter_id
    para_id
    embedding
    token
    text
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


LANGS = ("zh", "en")
EXPECTED_CHAPTER_IDS = set(range(67))


def warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_json(path):
    """讀取 JSON，支援 UTF-8 BOM。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到檔案: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_lines(path):
    """
    讀取 txt，並回傳每一行。
    使用 splitlines() 移除換行符號，但保留中間空行。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到文字檔: {path}")

    text = path.read_text(encoding="utf-8-sig")
    return text.splitlines()


def get_line_index(para_id, row_idx, line_count):
    """
    將 para_id 轉成文字檔行號：
        p1 -> 0
        p2 -> 1
        p3 -> 2
        ...

    若 para_id 不是 pN 格式，則改用 meta.json 中的順序 row_idx。
    若 pN 指向不存在的行，則報錯，避免靜默拿錯文字。
    """
    s = str(para_id).strip()

    if s.lower().startswith("p"):
        try:
            n = int(s[1:])
        except ValueError:
            # 例如 pX 這種無法解析的情況，退回 meta 順序
            if 0 <= row_idx < line_count:
                return row_idx
            raise IndexError(
                f"無法解析 para_id={para_id}，且後備行號 row_idx={row_idx} 超出範圍"
            )

        idx = n - 1

        if 0 <= idx < line_count:
            return idx

        raise IndexError(
            f"para_id={para_id} 指向第 {n} 行，但文字檔只有 {line_count} 行"
        )

    # 非 pN 格式時，使用 meta 順序作為後備行號
    if 0 <= row_idx < line_count:
        return row_idx

    raise IndexError(
        f"無法以 para_id={para_id} 取得行號，且後備行號 row_idx={row_idx} 超出範圍"
    )


def discover_chapter_ids(embeddings_dir):
    """
    從 embeddings 資料夾中找出所有數字命名的子資料夾。
    例如：0, 1, 2, ..., 66
    """
    ids = []

    for p in embeddings_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            ids.append(int(p.name))

    return sorted(ids)


def process_chapter(
    chapter_id,
    zh_dir,
    en_dir,
    tokens_dir,
    embeddings_dir,
    writers,
    chapter_id_as_int=False,
):
    """
    處理單一 chapter：
        embeddings/{chapter_id}/meta.json
        embeddings/{chapter_id}/zh.npy
        embeddings/{chapter_id}/en.npy
        tokens/{chapter_id}.json
        zh/ch{chapter_id}.txt
        en/ch{chapter_id}.txt
    """
    chapter_dir = embeddings_dir / str(chapter_id)
    meta_path = chapter_dir / "meta.json"
    token_path = tokens_dir / f"{chapter_id}.json"

    meta = load_json(meta_path)
    token_data = load_json(token_path)

    # 檢查 meta.json 的 chapter_id 是否與資料夾名稱一致
    meta_chapter_id = str(meta.get("chapter_id", chapter_id))
    if meta_chapter_id != str(chapter_id):
        raise ValueError(
            f"{meta_path} 的 chapter_id={meta_chapter_id!r} "
            f"與資料夾編號 {chapter_id} 不一致"
        )

    text_dir_map = {
        "zh": zh_dir,
        "en": en_dir,
    }

    chapter_id_value = int(chapter_id) if chapter_id_as_int else str(chapter_id)

    for lang in LANGS:
        para_ids_key = f"{lang}_para_ids"

        if para_ids_key not in meta:
            raise KeyError(f"{meta_path} 缺少欄位: {para_ids_key}")

        if not isinstance(meta[para_ids_key], list):
            raise ValueError(f"{meta_path} 的 {para_ids_key} 必須是陣列")

        para_ids = [str(x) for x in meta[para_ids_key]]

        # --------------------------------------------------
        # 讀取 embedding
        # --------------------------------------------------
        npy_path = chapter_dir / f"{lang}.npy"

        if not npy_path.is_file():
            raise FileNotFoundError(f"找不到 embedding 檔: {npy_path}")

        emb = np.load(npy_path, allow_pickle=False)

        if emb.ndim != 2:
            raise ValueError(
                f"{npy_path} 必須是二維陣列，例如 (num_paras, 1024)，"
                f"目前是 {emb.shape}"
            )

        if emb.shape[0] != len(para_ids):
            raise ValueError(
                f"{npy_path} 列數 {emb.shape[0]} "
                f"與 {para_ids_key} 數量 {len(para_ids)} 不一致"
            )

        if emb.shape[1] != 1024:
            warn(
                f"{npy_path} 的向量維度是 {emb.shape[1]}，"
                f"不是規範中的 1024。"
            )

        # --------------------------------------------------
        # 讀取 token 數
        # --------------------------------------------------
        if lang not in token_data:
            raise KeyError(f"{token_path} 缺少欄位: {lang}")

        lang_tokens = token_data[lang]

        if not isinstance(lang_tokens, dict):
            raise ValueError(f"{token_path} 的 {lang} 欄位必須是物件")

        # --------------------------------------------------
        # 讀取文字
        # --------------------------------------------------
        text_path = text_dir_map[lang] / f"ch{chapter_id}.txt"
        lines = read_lines(text_path)

        if len(lines) < len(para_ids):
            raise ValueError(
                f"{text_path} 只有 {len(lines)} 行，"
                f"但 {para_ids_key} 有 {len(para_ids)} 個段落"
            )

        # --------------------------------------------------
        # 逐段寫入
        # --------------------------------------------------
        for row_idx, para_id in enumerate(para_ids):
            if para_id not in lang_tokens:
                raise KeyError(
                    f"{token_path} 的 {lang} 缺少段落: {para_id}"
                )

            line_idx = get_line_index(
                para_id=para_id,
                row_idx=row_idx,
                line_count=len(lines),
            )

            record = {
                "chapter_id": chapter_id_value,
                "para_id": para_id,
                "embedding": emb[row_idx].tolist(),
                "token": lang_tokens[para_id],
                "text": lines[line_idx],
            }

            json_line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )

            writers[lang].write(json_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="合併 zh / en / tokens / embeddings 為 JSONL（.jsonl）檔案。"
    )

    parser.add_argument(
        "zh_dir",
        type=Path,
        help="zh 資料夾路徑，例如：./zh",
    )
    parser.add_argument(
        "en_dir",
        type=Path,
        help="en 資料夾路徑，例如：./en",
    )
    parser.add_argument(
        "tokens_dir",
        type=Path,
        help="tokens 資料夾路徑，例如：./tokens",
    )
    parser.add_argument(
        "embeddings_dir",
        type=Path,
        help="embeddings 資料夾路徑，例如：./embeddings 或 ./embedding",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="輸出資料夾，預設為目前資料夾",
    )

    parser.add_argument(
        "--chapter-id-type",
        choices=["str", "int"],
        default="str",
        help=(
            "chapter_id 輸出類型。"
            "預設為字串，例如 '0'；"
            "若希望輸出整數 0，可加 --chapter-id-type int"
        ),
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "允許缺少 0~66 中的某些章節。"
            "若不加此參數，只要缺少任何 0~66 章節就會報錯。"
        ),
    )

    args = parser.parse_args()

    # 檢查輸入資料夾
    for d in (
        args.zh_dir,
        args.en_dir,
        args.tokens_dir,
        args.embeddings_dir,
    ):
        if not d.is_dir():
            fail(f"輸入資料夾不存在或不是資料夾: {d}")

    # 找出 embeddings 中實際存在的章節
    found_all = set(discover_chapter_ids(args.embeddings_dir))

    if not found_all:
        fail(
            f"在 {args.embeddings_dir} 找不到任何數字命名的章節資料夾，"
            "例如 0、1、2 ..."
        )

    # 只處理 0~66；超出範圍的章節先忽略並警告
    missing = sorted(EXPECTED_CHAPTER_IDS - found_all)
    extra = sorted(found_all - EXPECTED_CHAPTER_IDS)

    if extra:
        warn(f"忽略超出 0~66 的章節資料夾: {extra}")

    if missing:
        msg = f"缺少 0~66 中的章節: {missing}"

        if args.allow_incomplete:
            warn(msg + "（已啟用 --allow-incomplete，只處理存在的章節）")
        else:
            fail(msg + "（若只想處理既有章節，請加 --allow-incomplete）")

    chapter_ids = sorted(EXPECTED_CHAPTER_IDS & found_all)

    if not chapter_ids:
        fail("沒有可處理的章節")

    # 建立輸出資料夾
    args.outdir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        lang: args.outdir / f"{lang}_paras.jsonl"
        for lang in LANGS
    }

    writers = {
        lang: out_paths[lang].open("w", encoding="utf-8")
        for lang in LANGS
    }

    try:
        for chapter_id in chapter_ids:
            process_chapter(
                chapter_id=chapter_id,
                zh_dir=args.zh_dir,
                en_dir=args.en_dir,
                tokens_dir=args.tokens_dir,
                embeddings_dir=args.embeddings_dir,
                writers=writers,
                chapter_id_as_int=(args.chapter_id_type == "int"),
            )
    except Exception as e:
        fail(f"處理失敗: {e}")
    finally:
        for fh in writers.values():
            fh.close()

    print(f"完成：{out_paths['zh']}")
    print(f"完成：{out_paths['en']}")


if __name__ == "__main__":
    main()