# -*- coding: utf-8 -*-
"""
02_embedding.py
--------------------------------------------------------------
對每個章節的中/英段落做 embedding。

改用 jinaai/jina-embeddings-v5-text-small（取代 v2-base-zh），
因為需要較新版 transformers，且 v5 的 encode() 需要指定 task。

本 pipeline 的用途是「比較中英段落語意是否相近」，是對稱式的相似度比對
（不是「用一句話去檢索文件」的不對稱檢索情境），所以預設使用
task="text-matching"（對應官方範例中的 Text Matching Task），
而不是 retrieval 的 query/document 兩段式 prompt。

輸出（供後續 04_align.py 使用）：
    /alignment_work/embeddings/{chapter_id}/zh.npy   (N_zh, dim)
    /alignment_work/embeddings/{chapter_id}/en.npy   (N_en, dim)
    /alignment_work/embeddings/{chapter_id}/meta.json
        {"chapter_id": ..., "task": ..., "zh_para_ids": [...], "en_para_ids": [...]}

獨立執行：python 02_embedding.py
"""

import os
import json
import argparse
import numpy as np

from common_io import CorpusLoader, ensure_dir


class ParagraphEmbedder:
    """
    封裝 jina-embeddings-v5-text-small，批次編碼段落，供跨語言相似度比對使用。

    GPU 上預設啟用 bfloat16 + flash_attention_2（官方建議設定）；
    若目前環境沒有裝 flash-attn，會自動 fallback 成預設 attention 實作並印出警告。
    CPU 環境則不套用 bfloat16 / flash-attention，避免相容性問題。
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v5-text-small",
        batch_size: int = 32,
        device: str = None,
        task: str = "text-matching",
        truncate_dim: int = None,
        use_flash_attention: bool = True,
        max_seq_length: int = None,
    ):
        """
        max_seq_length: 若設定，會截斷過長輸入的 token 數再送進模型。
            04_align.py 在做 vecalign overlap 編碼時，單一輸入可能是多個段落
            拼接起來的長文字（alignment_max_size 越大，最長可到 8 段拼接），
            若不設上限，即使 batch_size=1 也可能因單一樣本序列過長、
            attention 記憶體暴增而 OOM。建議跑 04_align.py 時明確帶入此參數。
        """
        from sentence_transformers import SentenceTransformer
        import torch

        self.torch = torch
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.task = task
        self.truncate_dim = truncate_dim

        model_kwargs = {}
        config_kwargs = {}
        if self.device.type == "cuda":
            model_kwargs["dtype"] = torch.bfloat16  # 官方建議在 GPU 上使用
            if use_flash_attention:
                config_kwargs["_attn_implementation"] = "flash_attention_2"

        print(f"[ParagraphEmbedder] 載入模型 {model_name} 於 {self.device} (task={task}) ...")
        try:
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=str(self.device),
                model_kwargs=model_kwargs,
                config_kwargs=config_kwargs,
            )
        except Exception as e:
            if config_kwargs.get("_attn_implementation") == "flash_attention_2":
                print(f"[WARN] flash_attention_2 載入失敗（{e}），改用預設 attention 實作重試")
                config_kwargs.pop("_attn_implementation", None)
                self.model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    device=str(self.device),
                    model_kwargs=model_kwargs,
                    config_kwargs=config_kwargs,
                )
            else:
                raise

        if max_seq_length:
            self.model.max_seq_length = max_seq_length
            print(f"[ParagraphEmbedder] max_seq_length 設為 {max_seq_length}（超過會被截斷）")

    def empty_cache(self):
        """釋放 PyTorch caching allocator 保留但目前未使用的 GPU 記憶體，緩解長時間跑多批次後的碎片化。"""
        import gc

        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def encode(self, texts, task: str = None, prompt_name: str = None):
        """
        task: 覆蓋預設 task（例如需要做 retrieval 時傳入 "retrieval"）。
        prompt_name: 搭配 retrieval 使用，"query" 或 "document"；
                     text-matching 不需要 prompt_name。
        """
        if not texts:
            dim = self.truncate_dim or self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,  # 方便直接用內積當 cosine similarity
            "task": task or self.task,
        }
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name
        if self.truncate_dim:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        embs = self.model.encode(texts, **encode_kwargs)
        return np.asarray(embs, dtype=np.float32)


class ChapterEmbeddingRunner:
    """跑遍所有章節，輸出 embedding 檔案。"""

    def __init__(
        self,
        zh_dir: str = "/output_zh",
        en_dir: str = "/output_en_utf8",
        out_dir: str = "/alignment_work/embeddings",
        embedder: ParagraphEmbedder = None,
    ):
        self.loader = CorpusLoader(zh_dir=zh_dir, en_dir=en_dir)
        self.out_dir = out_dir
        self.embedder = embedder or ParagraphEmbedder()
        ensure_dir(self.out_dir)

    def run(self):
        indices = self.loader.list_chapter_indices()
        print(f"[INFO] 共 {len(indices)} 個章節待編碼")
        for idx in indices:
            chapter = self.loader.load_chapter(idx)
            chap_out = os.path.join(self.out_dir, chapter.chapter_id)
            ensure_dir(chap_out)

            zh_emb = self.embedder.encode(chapter.zh_paragraphs)
            en_emb = self.embedder.encode(chapter.en_paragraphs)

            np.save(os.path.join(chap_out, "zh.npy"), zh_emb)
            np.save(os.path.join(chap_out, "en.npy"), en_emb)

            with open(os.path.join(chap_out, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chapter_id": chapter.chapter_id,
                        "task": self.embedder.task,
                        "zh_para_ids": chapter.zh_para_ids,
                        "en_para_ids": chapter.en_para_ids,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"  - {chapter.chapter_id}: zh={len(chapter.zh_paragraphs)} en={len(chapter.en_paragraphs)} 段落已編碼")

        print(f"[DONE] embedding 輸出於 {self.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="對中英段落做 jina-embeddings-v5-text-small embedding")
    parser.add_argument("--zh_dir", default="/output_zh")
    parser.add_argument("--en_dir", default="/output_en_utf8", help="建議先跑過 01_unicode_transform.py")
    parser.add_argument("--out_dir", default="/alignment_work/embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", default="jinaai/jina-embeddings-v5-text-small")
    parser.add_argument("--task", default="text-matching", choices=["text-matching", "retrieval"])
    parser.add_argument("--truncate_dim", type=int, default=None, help="可選，縮短向量維度以省記憶體")
    parser.add_argument("--max_seq_length", type=int, default=None, help="可選，截斷輸入 token 數上限，避免長文字 OOM")
    parser.add_argument("--no_flash_attention", action="store_true", help="停用 flash_attention_2")
    args = parser.parse_args()

    embedder = ParagraphEmbedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        task=args.task,
        truncate_dim=args.truncate_dim,
        max_seq_length=args.max_seq_length,
        use_flash_attention=not args.no_flash_attention,
    )
    runner = ChapterEmbeddingRunner(
        zh_dir=args.zh_dir, en_dir=args.en_dir, out_dir=args.out_dir, embedder=embedder
    )
    runner.run()


if __name__ == "__main__":
    main()