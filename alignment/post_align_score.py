# -*- coding: utf-8 -*-
"""
05_recalc_scores.py
針對 post_align 後 realigned=True 的紀錄，重新計算 similarity 與 vecalign_score。
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from importlib import util as _il_util

def load_module(mod_path, mod_name):
    spec = _il_util.spec_from_file_location(mod_name, mod_path)
    mod = _il_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def calculate_similarity(embedder, zh_text, en_text):
    """
    使用 torch.nn.functional.cosine_similarity 計算 cosine similarity。
    """
    if not zh_text or not en_text:
        return 0.0
        
    emb = embedder.encode([zh_text, en_text])
    
    if isinstance(emb, torch.Tensor):
        emb_t = emb.detach().cpu().to(dtype=torch.float32)
    else:
        emb_t = torch.as_tensor(np.asarray(emb, dtype=np.float32))

    if emb_t.ndim != 2 or emb_t.shape[0] < 2:
        return 0.0

    sim = F.cosine_similarity(
        emb_t[0].unsqueeze(0),
        emb_t[1].unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).item()

    return float(sim) if np.isfinite(sim) else 0.0

def main():
    parser = argparse.ArgumentParser(description="重新計算 post_align 後 realigned=True 的 similarity 與 vecalign_score")
    parser.add_argument("--input_jsonl", required=True, help="post_align 輸出的 JSONL 檔案路徑")
    parser.add_argument("--output_jsonl", required=True, help="重新計算分數後的輸出 JSONL 檔案路徑")
    
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--embed_max_seq_length", type=int, default=1024)
    
    args = parser.parse_args()

    # 動態載入 02_embedding.py
    here = os.path.dirname(os.path.abspath(__file__))
    emb_mod_path = os.path.join(here, "02_embedding.py")
    if not os.path.exists(emb_mod_path):
        emb_mod_path = "02_embedding.py" # Fallback
        
    if not os.path.exists(emb_mod_path):
        raise FileNotFoundError(f"找不到 02_embedding.py，請確保它與本腳本在同一目錄或指定正確路徑。")

    print(f"[INFO] 載入 {emb_mod_path} ...")
    emb_mod = load_module(emb_mod_path, "embedding_mod")
    
    embedder = emb_mod.ParagraphEmbedder(
        batch_size=args.embed_batch_size,
        max_seq_length=args.embed_max_seq_length,
    )

    updated_count = 0
    total_count = 0
    
    print(f"[INFO] 開始處理 {args.input_jsonl} ...")
    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            total_count += 1
            
            # 只處理 realigned == True 的紀錄
            if record.get("realigned", False):
                zh_text = record.get("zh_text", "")
                en_text = record.get("en_text", "")
                
                # 1. 計算新的 similarity
                new_sim = calculate_similarity(embedder, zh_text, en_text)
                record["similarity"] = round(new_sim, 4)
                
                # 2. 處理 vecalign_score
                # 由於無法針對單一 pair 重跑全局 DP，這裡將其更新為新的 similarity 作為近似值。
                # 如果你希望保留原始的全域 vecalign_score，請將下方這行註解掉。
                record["vecalign_score"] = round(new_sim, 4) 
                
                updated_count += 1
                
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] 總共處理 {total_count} 筆，更新了 {updated_count} 筆 realigned 紀錄。")
    print(f"[DONE] 輸出至 {args.output_jsonl}")

if __name__ == "__main__":
    main()