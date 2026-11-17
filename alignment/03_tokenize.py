# -*- coding: utf-8 -*-
"""
03_tokenize.py
--------------------------------------------------------------
用目標微調模型 Qwen/Qwen3-4B 的 tokenizer 對所有段落做 token 計數，
用來衡量段落長度（後續用於切分 / 統計 / outlier 判斷）。

注意：這裡只需要 tokenizer，不需要載入完整模型權重，
      可大幅節省 Colab T4 的記憶體與時間。

輸出：
    /alignment_work/tokens/{chapter_id}.json
        {
          "zh": {"p1": 12, "p2": 34, ...},
          "en": {"p1": 10, "p2": 20, ...}
        }

獨立執行：python 03_tokenize.py
"""

import os
import json
import argparse

from common_io import CorpusLoader, ensure_dir


class ParagraphTokenizer:
    """封裝 Qwen3-4B tokenizer，僅用於計算 token 數，不載入模型權重。"""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        from transformers import AutoTokenizer

        print(f"[ParagraphTokenizer] 載入 tokenizer {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens_batch(self, texts):
        if not texts:
            return []
        # 批次 encode，不加特殊 token，純粹量測長度
        enc = self.tokenizer(texts, add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]


class ChapterTokenCounter:
    def __init__(
        self,
        zh_dir: str = "/output_zh",
        en_dir: str = "/output_en_utf8",
        out_dir: str = "/alignment_work/tokens",
        tokenizer: ParagraphTokenizer = None,
    ):
        self.loader = CorpusLoader(zh_dir=zh_dir, en_dir=en_dir)
        self.out_dir = out_dir
        self.tokenizer = tokenizer or ParagraphTokenizer()
        ensure_dir(self.out_dir)

    def run(self):
        indices = self.loader.list_chapter_indices()
        print(f"[INFO] 共 {len(indices)} 個章節待計算 token 數")
        for idx in indices:
            chapter = self.loader.load_chapter(idx)

            zh_counts = self.tokenizer.count_tokens_batch(chapter.zh_paragraphs)
            en_counts = self.tokenizer.count_tokens_batch(chapter.en_paragraphs)

            result = {
                "zh": dict(zip(chapter.zh_para_ids, zh_counts)),
                "en": dict(zip(chapter.en_para_ids, en_counts)),
            }

            out_path = os.path.join(self.out_dir, f"{chapter.chapter_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  - {chapter.chapter_id}: zh_avg={self._avg(zh_counts):.1f} en_avg={self._avg(en_counts):.1f}")

        print(f"[DONE] token 計數輸出於 {self.out_dir}")

    @staticmethod
    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0


def main():
    parser = argparse.ArgumentParser(description="用 Qwen3-4B tokenizer 計算段落 token 數")
    parser.add_argument("--zh_dir", default="/output_zh")
    parser.add_argument("--en_dir", default="/output_en_utf8")
    parser.add_argument("--out_dir", default="/alignment_work/tokens")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    tokenizer = ParagraphTokenizer(model_name=args.model_name)
    counter = ChapterTokenCounter(
        zh_dir=args.zh_dir, en_dir=args.en_dir, out_dir=args.out_dir, tokenizer=tokenizer
    )
    counter.run()


if __name__ == "__main__":
    main()
