#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_alignment.py
驗證 zh/en txt、tokens、embeddings 四個資料夾的行數/段落數是否完全對齊。
用法: python verify_alignment.py ./zh ./en ./tokens ./embeddings
"""

import json
import sys
from pathlib import Path
import numpy as np

LANGS = ("zh", "en")
EXPECTED_CHAPTERS = range(67)


def verify(zh_dir, en_dir, tokens_dir, emb_dir):
    text_dirs = {"zh": zh_dir, "en": en_dir}
    results = []       # [(chapter, lang, txt_lines, token_keys, emb_rows, meta_ids, status)]
    all_ok = True

    for ch in EXPECTED_CHAPTERS:
        meta_path = emb_dir / str(ch) / "meta.json"
        token_path = tokens_dir / f"{ch}.json"

        # 載入 meta & token（若缺失則標記錯誤）
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            token_data = json.loads(token_path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            for lang in LANGS:
                results.append((ch, lang, None, None, None, None, f"LOAD_ERROR: {e}"))
            all_ok = False
            continue

        for lang in LANGS:
            para_ids_key = f"{lang}_para_ids"
            meta_ids = meta.get(para_ids_key, [])
            meta_count = len(meta_ids)

            # --- txt 行數 ---
            txt_path = text_dirs[lang] / f"ch{ch}.txt"
            try:
                txt_lines = len(txt_path.read_text(encoding="utf-8-sig").splitlines())
            except FileNotFoundError:
                txt_lines = None

            # --- token key 數量 ---
            lang_tokens = token_data.get(lang, {})
            # 只計算 meta 中有列出的 para_id 是否存在於 token 中
            matched_token_keys = sum(1 for pid in meta_ids if pid in lang_tokens)
            missing_tokens = [pid for pid in meta_ids if pid not in lang_tokens]

            # --- embedding 列數 ---
            npy_path = emb_dir / str(ch) / f"{lang}.npy"
            try:
                emb = np.load(npy_path, allow_pickle=False)
                emb_rows = emb.shape[0] if emb.ndim == 2 else None
            except Exception:
                emb_rows = None

            # --- 判定狀態 ---
            issues = []
            if txt_lines is not None and txt_lines < meta_count:
                issues.append(f"txt行數({txt_lines})<meta({meta_count})")
            if matched_token_keys != meta_count:
                issues.append(f"token缺{len(missing_tokens)}個key:{missing_tokens[:3]}...")
            if emb_rows is not None and emb_rows != meta_count:
                issues.append(f"emb列數({emb_rows})!=meta({meta_count})")
            if txt_lines is None:
                issues.append("txt檔案不存在")
            if emb_rows is None:
                issues.append("npy檔案不存在或讀取失敗")

            status = "OK" if not issues else "; ".join(issues)
            if status != "OK":
                all_ok = False

            results.append((ch, lang, txt_lines, matched_token_keys, emb_rows, meta_count, status))

    # --- 列印報告 ---
    header = f"{'Ch':>3} {'Lang':>4} {'TxtLines':>8} {'TokenKeys':>9} {'EmbRows':>8} {'MetaIDs':>8} Status"
    print(header)
    print("-" * len(header.encode('utf-8')))

    totals = {lang: {"txt": 0, "token": 0, "emb": 0, "meta": 0} for lang in LANGS}
    error_count = 0

    for ch, lang, txt_l, tok_k, emb_r, meta_c, status in results:
        display_txt = str(txt_l) if txt_l is not None else "N/A"
        display_tok = str(tok_k) if tok_k is not None else "N/A"
        display_emb = str(emb_r) if emb_r is not None else "N/A"
        display_meta = str(meta_c) if meta_c is not None else "N/A"

        print(f"{ch:>3} {lang:>4} {display_txt:>8} {display_tok:>9} {display_emb:>8} {display_meta:>8} {status}")

        if status == "OK":
            totals[lang]["txt"] += txt_l or 0
            totals[lang]["token"] += tok_k or 0
            totals[lang]["emb"] += emb_r or 0
            totals[lang]["meta"] += meta_c or 0
        else:
            error_count += 1

    # --- 總計摘要 ---
    print("\n" + "=" * 70)
    print("【總計摘要】")
    for lang in LANGS:
        t = totals[lang]
        aligned = (t["txt"] == t["token"] == t["emb"] == t["meta"])
        symbol = "✅" if aligned else "❌"
        print(f"  {symbol} {lang}: txt={t['txt']}, token={t['token']}, emb={t['emb']}, meta={t['meta']}")

    if error_count > 0:
        print(f"\n⚠️  共有 {error_count} 筆對齊異常，請先修復再執行合併。")
        sys.exit(1)
    else:
        print("\n✅ 所有章節的四個來源行數完全對齊，可以安全執行合併。")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python verify_alignment.py <zh_dir> <en_dir> <tokens_dir> <embeddings_dir>")
        sys.exit(1)

    verify(
        zh_dir=Path(sys.argv[1]),
        en_dir=Path(sys.argv[2]),
        tokens_dir=Path(sys.argv[3]),
        emb_dir=Path(sys.argv[4]),
    )