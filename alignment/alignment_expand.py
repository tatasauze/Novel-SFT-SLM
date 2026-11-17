# -- coding: utf-8 --
"""
04_align.py
段落層級對齊。
本版改為直接呼叫「真正的 vecalign」(thompsonb/vecalign)，
以支援 --alignment_max_size 8（可涵蓋單側最多約 6-7 句合併的對齊情境）。
前置需求（Colab 上先跑一次）：
!git clone https://github.com/thompsonb/vecalign.git /content/vecalign
對每個章節，把該章的中/英段落文字直接交給 vecalign：
1. overlap.py 產生「連續 1~N 行」的所有重疊組合文字 (overlaps)
2. 用 02_embedding.py 的 jina embedder 對 overlaps 逐行編碼，
輸出成 vecalign 需要的 raw float32 binary (.emb)
3. 呼叫 vecalign.py 做全域對齊，取得 (src_idxs, tgt_idxs, score)
輸出 alignment_unit = "paragraph_pair" 的 JSONL。
對齊失敗（某一側完全沒有對應，即 vecalign 判定為刪除/插入）的段落，
不再被直接捨棄，而是輸出到 paragraph_unaligned.jsonl 供人工檢視。
輸出：
/alignment_work/results/paragraph_align.jsonl
/alignment_work/results/paragraph_unaligned.jsonl   # 對齊失敗，供人工檢視
獨立執行：python 04_align.py --vecalign_dir /content/vecalign
"""
import os
import re
import ast
import sys
import gc
import json
import uuid
import argparse
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from common_io import CorpusLoader, ensure_dir, write_jsonl

# ------------------------------------------------------------------
# vecalign 封裝：呼叫真正的 overlap.py + vecalign.py
# ------------------------------------------------------------------
class VecalignRunner:
    """
    封裝 thompsonb/vecalign 的完整流程：
    1. 把 zh_lines / en_lines 寫成暫存文字檔（一行一單位）
    2. 呼叫 vecalign 的 overlap.py 產生 overlap 文本
    （即所有長度 1..alignment_max_size 的連續行拼接組合）
    3. 用自帶的 embedder（jina embeddings）對 overlap 文本逐行編碼，
    輸出成 vecalign 需要的 raw float32 binary
    4. 呼叫 vecalign.py 進行全域對齊，解析 stdout 取得對齊結果
    回傳的每筆對齊為 (src_idxs, tgt_idxs, vecalign_score)，
     src_idxs / tgt_idxs 為 0-based 索引 list，任一方為空代表該側無對應（刪除）。
     """

    def __init__(
        self,
        embedder,
        vecalign_dir: str = "/content/vecalign",
        alignment_max_size: int = 8,
        work_dir: str = "/alignment_work/vecalign_tmp",
        vecalign_extra_args: dict = None, # 新增：支援傳入 vecalign 的其他 CLI 參數
    ):
        self.embedder = embedder
        self.alignment_max_size = alignment_max_size
        self.work_dir = work_dir
        self.vecalign_extra_args = vecalign_extra_args or {}
        ensure_dir(self.work_dir)
        
        # 有些人會不小心在已經是 /content/vecalign 的目錄下再 clone 一次
        # (git clone ... 沒指定目的地時，會依 repo 名稱建立子資料夾)，
        # 導致實際檔案跑到 /content/vecalign/vecalign/ 底下。這裡自動偵測兩種可能位置。
        candidates = [vecalign_dir, os.path.join(vecalign_dir, "vecalign")]
        resolved_dir = None
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "overlap.py")) and os.path.exists(
                os.path.join(cand, "vecalign.py")
            ):
                resolved_dir = cand
                break
        if resolved_dir is None:
            checked = "\n".join(f"  - {c}" for c in candidates)
            raise FileNotFoundError(
                f"在下列位置都找不到 overlap.py / vecalign.py：\n{checked}\n"
                f"請確認已用『絕對路徑』乾淨 clone 一次官方 repo（避免巢狀目錄）：\n"
                f"  rm -rf {vecalign_dir}\n"
                f"  git clone https://github.com/thompsonb/vecalign.git {vecalign_dir}"
            )
        self.vecalign_dir = resolved_dir
        
        # repo 根目錄有 __init__.py，代表 overlap.py / vecalign.py / dp_utils.py 之間
        # 是用「相對匯入」(from .dp_utils import ...) 寫的，必須以 package 模式執行
        self.package_parent = os.path.dirname(os.path.abspath(resolved_dir))
        self.package_name = os.path.basename(os.path.abspath(resolved_dir))
        if not os.path.exists(os.path.join(resolved_dir, "__init__.py")):
            print(
                f"[WARN] {resolved_dir} 底下沒有 __init__.py，"
                f"可能不是完整的 repo（例如 clone 中斷），以 package 模式執行可能仍會失敗。"
            )
            
        self._ensure_dp_core_available()
        stray_cpp = [
            f for f in os.listdir(resolved_dir)
            if f.endswith((".cpp", ".c", ".so")) and not f.startswith("dp_core")
        ]
        if stray_cpp:
            print(
                f"[WARN] {resolved_dir} 內發現非官方 repo 自帶的編譯產物/手動建立的檔案: {stray_cpp}\n"
                f"       如果曾手動建立過空的 .cpp 檔案來『解決』build 錯誤，"
                f"請刪除它們並重新乾淨 clone，否則可能讓 pyximport 誤判已經編譯完成而載入到空的/損壞的模組。"
            )

    def _dp_core_importable(self):
        probe = (
            f"import sys; sys.path.insert(0, {self.package_parent!r}); "
            f"import {self.package_name}.dp_core"
        )
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        return r.returncode == 0, r.stderr

    def _ensure_dp_core_available(self):
        ok, err = self._dp_core_importable()
        if ok:
            return
        print(
            "[INFO] vecalign 的 dp_core（Cython 編譯的 DP 加速模組）目前無法匯入，"
            "嘗試自動編譯 ..."
        )
        # 預防性安裝：確保編譯 Cython 擴充模組所需的基礎依賴存在
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "cython", "numpy"],
            capture_output=True, text=True
        )
        # 嘗試 1：pip install -e（可編輯模式安裝）
        install_candidates = []
        for cand in (self.package_parent, self.vecalign_dir):
            if cand not in install_candidates:
                install_candidates.append(cand)
        for install_target in install_candidates:
            has_build_file = any(
                os.path.exists(os.path.join(install_target, f))
                for f in ("setup.py", "pyproject.toml")
            )
            if not has_build_file:
                continue
            print(f"[INFO] 偵測到 {install_target} 有 setup.py/pyproject.toml，嘗試 pip install -e ...")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", install_target],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"[WARN] pip install -e {install_target} 失敗：\n{r.stderr[-1500:]}")
                continue
            ok, err = self._dp_core_importable()
            if ok:
                print("[INFO] pip install -e 後 dp_core 已可正常匯入")
                return
                
        # 嘗試 2：使用 setup.py 進行原地編譯 (build_ext --inplace)
        setup_py_path = os.path.join(self.vecalign_dir, "setup.py")
        if os.path.exists(setup_py_path):
            print(f"[INFO] 嘗試使用 setup.py 進行原地編譯 (build_ext --inplace) ...")
            r = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                capture_output=True, text=True, 
                cwd=self.vecalign_dir  # 必須在 setup.py 所在目錄執行
            )
            if r.returncode == 0:
                ok, err = self._dp_core_importable()
                if ok:
                    print("[INFO] 原地編譯 dp_core 成功")
                    return
            else:
                err = r.stderr
                print(f"[WARN] setup.py build_ext --inplace 失敗：\n{err[-1500:]}")
                
        raise RuntimeError(
            f"[VecalignRunner] 無法讓 {self.package_name}.dp_core 可以被匯入，"
            f"已嘗試 pip install -e 與 setup.py build_ext --inplace 皆失敗。"
            f"這通常代表環境缺少 C 編譯工具鏈（gcc/build-essential）或 cython 版本不相容，"
            f"而不是我們 wrapper 程式碼的問題。請在 Colab 手動執行以下指令觀察實際錯誤：\n"
            f"  apt-get install build-essential\n"
            f"  cd {self.vecalign_dir} && python setup.py build_ext --inplace\n"
            f"--- 最後一次錯誤訊息 ---\n{err}"
        )

    @staticmethod
    def _write_lines(path, lines):
        # vecalign 以行為單位，段落/句子內部若含換行需先攤平，避免破壞行對應關係
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ln.replace("\n", " ").replace("\r", " ") for ln in lines) + "\n")

    def _run_module(self, module_name, cmd_args, step_name):
        """
        以 `python -m {package_name}.{module_name}` 執行，並把 package 的上層目錄
        加進 PYTHONPATH / 設成 cwd，讓 repo 內的相對匯入 (from .dp_utils import ...) 能正常解析。
        """
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = self.package_parent + (os.pathsep + existing if existing else "")
        cmd = [sys.executable, "-m", f"{self.package_name}.{module_name}"] + cmd_args
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=self.package_parent, env=env
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[VecalignRunner] {step_name} 執行失敗 (exit code {result.returncode})。\n"
                f"這是 vecalign repo 自己的模組丟出的錯誤，不是我們 wrapper 程式碼本身的例外，"
                f"請依下面的原始錯誤訊息判斷（常見原因：cython/dp_core.pyx 編譯失敗、"
                f"輸入檔案為空、或 vecalign_dir 指到錯的路徑）。\n"
                f"--- command (cwd={self.package_parent}) ---\n{' '.join(cmd)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n"
            )
        return result.stdout

    def _make_overlaps(self, txt_path, overlap_path):
        self._run_module(
            "overlap",
            ["-i", txt_path, "-o", overlap_path, "-n", str(self.alignment_max_size)],
            step_name="overlap.py",
        )

    def _embed_overlaps(self, overlap_path, emb_path):
        with open(overlap_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        emb = self.embedder.encode(lines)
        emb.astype(np.float32).tofile(emb_path)
        # 每次 encode 後主動釋放 PyTorch caching allocator 保留但未使用的記憶體。
        if hasattr(self.embedder, "empty_cache"):
            self.embedder.empty_cache()

    @staticmethod
    def _parse_output(stdout: str):
        """
        vecalign.py 輸出格式（每行一筆對齊）：
            [0]:[0]:0.156006
            [6]:[6, 7, 8]:0.507506
            []:[22]:0.000000          # 插入/刪除（某一側為空）
        """
        alignments = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                src_str, tgt_str, score_str = line.rsplit(":", 2)
                src_idxs = ast.literal_eval(src_str)
                tgt_idxs = ast.literal_eval(tgt_str)
                score = float(score_str)
            except Exception:
                print(f"[WARN] 無法解析 vecalign 輸出行，已略過: {line!r}")
                continue
            alignments.append((list(src_idxs), list(tgt_idxs), score))
        return alignments

    def align(self, zh_lines, en_lines, tag="pair"):
        if not zh_lines or not en_lines:
            return []
        uid = uuid.uuid4().hex[:8]
        base = os.path.join(self.work_dir, f"{tag}_{uid}")
        src_txt, tgt_txt = base + ".zh.txt", base + ".en.txt"
        src_ov, tgt_ov = base + ".zh.overlap", base + ".en.overlap"
        src_emb, tgt_emb = src_ov + ".emb", tgt_ov + ".emb"
        try:
            self._write_lines(src_txt, zh_lines)
            self._write_lines(tgt_txt, en_lines)
            self._make_overlaps(src_txt, src_ov)
            self._make_overlaps(tgt_txt, tgt_ov)
            self._embed_overlaps(src_ov, src_emb)
            self._embed_overlaps(tgt_ov, tgt_emb)
            
            # 組裝 vecalign.py 的 CLI 參數
            cmd_args = [
                "--alignment_max_size", str(self.alignment_max_size),
                "--src", src_txt,
                "--tgt", tgt_txt,
                "--src_embed", src_ov, src_emb,
                "--tgt_embed", tgt_ov, tgt_emb,
            ]
            
            # 動態加入額外傳入的 vecalign 參數
            for k, v in self.vecalign_extra_args.items():
                if isinstance(v, bool):
                    if v:
                        cmd_args.append(f"--{k}")
                else:
                    cmd_args.extend([f"--{k}", str(v)])

            stdout = self._run_module(
                "vecalign",
                cmd_args,
                step_name="vecalign.py",
            )
            return self._parse_output(stdout)
        finally:
            for p in [src_txt, tgt_txt, src_ov, tgt_ov, src_emb, tgt_emb]:
                try:
                    os.remove(p)
                except OSError:
                    pass

# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
class AlignmentPipeline:
    def __init__(
        self,
        zh_dir="/output_zh",
        en_dir="/output_en_utf8",
        tok_dir="/alignment_work/tokens",
        out_dir="/alignment_work/results",
        embedder=None,
        vecalign_dir="/content/vecalign",
        alignment_max_size=8,
        vecalign_extra_args=None, # 新增
    ):
        self.loader = CorpusLoader(zh_dir=zh_dir, en_dir=en_dir)
        self.tok_dir = tok_dir
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self._embedder = embedder
        self.alignment_max_size = alignment_max_size
        self._vecalign = None
        self._vecalign_dir = vecalign_dir
        self.vecalign_extra_args = vecalign_extra_args or {}

    def set_embedder(self, embedder):
        self._embedder = embedder

    @property
    def vecalign(self):
        if self._vecalign is None:
            if self._embedder is None:
                raise RuntimeError("需要先呼叫 set_embedder()")
            self._vecalign = VecalignRunner(
                embedder=self._embedder,
                vecalign_dir=self._vecalign_dir,
                alignment_max_size=self.alignment_max_size,
                vecalign_extra_args=self.vecalign_extra_args, # 傳遞參數
            )
        return self._vecalign

    def _load_tokens(self, chapter_id):
        tok_path = os.path.join(self.tok_dir, f"{chapter_id}.json")
        with open(tok_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _similarity(self, zh_text, en_text):
        """
        使用 torch.nn.functional.cosine_similarity 計算 cosine similarity。
        """
        emb = self._embedder.encode([zh_text, en_text])
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
        sim = float(sim)
        if not np.isfinite(sim):
            sim = 0.0
        return sim

    def run(self):
        indices = self.loader.list_chapter_indices()
        paragraph_records = []
        paragraph_unaligned_records = []  
        for idx in indices:
            chapter = self.loader.load_chapter(idx)
            tok_data = self._load_tokens(chapter.chapter_id)
            alignments = self.vecalign.align(
                chapter.zh_paragraphs,
                chapter.en_paragraphs,
                tag=f"{chapter.chapter_id}_para",
            )
            for zh_idxs, en_idxs, score in alignments:
                if not zh_idxs or not en_idxs:
                    self._record_unaligned(
                        target_list=paragraph_unaligned_records,
                        chapter=chapter,
                        tok_data=tok_data,
                        zh_idxs=zh_idxs,
                        en_idxs=en_idxs,
                    )
                    continue
                zh_para_ids = [chapter.zh_para_ids[k] for k in zh_idxs]
                en_para_ids = [chapter.en_para_ids[k] for k in en_idxs]
                zh_text = "".join(chapter.zh_paragraphs[k] for k in zh_idxs)
                en_text = " ".join(chapter.en_paragraphs[k] for k in en_idxs)
                zh_tok = sum(tok_data["zh"][pid] for pid in zh_para_ids)
                en_tok = sum(tok_data["en"][pid] for pid in en_para_ids)
                sim = self._similarity(zh_text, en_text)
                record = {
                    "chapter_id": chapter.chapter_id,
                    "parent_en_para_ids": en_para_ids,
                    "parent_zh_para_ids": zh_para_ids,
                    "en_text": en_text,
                    "zh_text": zh_text,
                    "en_tokens": en_tok,
                    "zh_tokens": zh_tok,
                    "similarity": round(sim, 4),
                    "vecalign_score": float(score),
                }
                paragraph_records.append(record)
            print(
                f"[INFO] {chapter.chapter_id} 完成，"
                f"累積 paragraph_pair={len(paragraph_records)}，"
                f"paragraph_unaligned={len(paragraph_unaligned_records)}"
            )
            if hasattr(self._embedder, "empty_cache"):
                self._embedder.empty_cache()
                
        write_jsonl(
            os.path.join(self.out_dir, "paragraph_align.jsonl"),
            paragraph_records,
        )
        write_jsonl(
            os.path.join(self.out_dir, "paragraph_unaligned.jsonl"),
            paragraph_unaligned_records,
        )
        print(f"[DONE] paragraph_align.jsonl: {len(paragraph_records)} 筆")
        print(f"[DONE] paragraph_unaligned.jsonl: {len(paragraph_unaligned_records)} 筆")

    @staticmethod
    def _record_unaligned(target_list, chapter, tok_data, zh_idxs, en_idxs):
        if zh_idxs and not en_idxs:
            side = "zh_only"
            para_ids = [chapter.zh_para_ids[k] for k in zh_idxs]
            text = "".join(chapter.zh_paragraphs[k] for k in zh_idxs)
            tokens = sum(tok_data["zh"].get(pid, 0) for pid in para_ids)
        elif en_idxs and not zh_idxs:
            side = "en_only"
            para_ids = [chapter.en_para_ids[k] for k in en_idxs]
            text = " ".join(chapter.en_paragraphs[k] for k in en_idxs)
            tokens = sum(tok_data["en"].get(pid, 0) for pid in para_ids)
        else:
            return  
        target_list.append({
            "chapter_id": chapter.chapter_id,
            "side": side,
            "para_ids": para_ids,
            "text": text,
            "tokens": tokens,
            "reason": "vecalign_no_counterpart",
        })

def main():
    parser = argparse.ArgumentParser(
        description="用 vecalign 做段落層級對齊（不做 chunk 對齊）"
    )
    parser.add_argument("--zh_dir", default="/output_zh")
    parser.add_argument("--en_dir", default="/output_en_utf8")
    parser.add_argument("--tok_dir", default="/alignment_work/tokens")
    parser.add_argument("--out_dir", default="/alignment_work/results")
    parser.add_argument(
        "--vecalign_dir",
        default="/content/vecalign",
        help="git clone https://github.com/thompsonb/vecalign.git 的路徑",
    )
    parser.add_argument("--alignment_max_size", type=int, default=8)
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=16,
        help="embedding batch size",
    )
    parser.add_argument(
        "--embed_max_seq_length",
        type=int,
        default=1024,
        help="embedding 輸入 token 數上限（截斷）",
    )
    
    # --- 新增 Vecalign 官方支援的其他 CLI 參數 ---
    parser.add_argument(
        "--one_to_many", type=int, nargs="?", const=50, default=None,
        help="One-to-many mode with optional M limit (default 50 if flag is used without value)"
    )
    parser.add_argument("--del_percentile_frac", type=float, default=None, help="Deletion penalty percentile")
    parser.add_argument("--max_size_full_dp", type=int, default=None, help="Threshold for full DP")
    parser.add_argument("--costs_sample_size", type=int, default=None, help="Cost estimation samples")
    parser.add_argument("--num_samps_for_norm", type=int, default=None, help="Normalization samples")
    parser.add_argument("--search_buffer_size", type=int, default=None, help="Search buffer width")
    parser.add_argument("--print_aligned_text", action="store_true", help="Print aligned text for debugging")
    parser.add_argument("--debug_save_stack", type=str, default=None, help="Save internal state to pickle")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    args = parser.parse_args()

    # 收集非 None 的額外 vecalign 參數
    vecalign_extra_args = {}
    for key in [
        "one_to_many", "del_percentile_frac", "max_size_full_dp", 
        "costs_sample_size", "num_samps_for_norm", "search_buffer_size", "debug_save_stack"
    ]:
        val = getattr(args, key, None)
        if val is not None:
            vecalign_extra_args[key] = val
            
    if getattr(args, "print_aligned_text", False):
        vecalign_extra_args["print_aligned_text"] = True
    if getattr(args, "verbose", False):
        vecalign_extra_args["verbose"] = True

    from importlib import util as _il_util
    def _load_module(mod_path, mod_name):
        spec = _il_util.spec_from_file_location(mod_name, mod_path)
        mod = _il_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    here = os.path.dirname(os.path.abspath(__file__))
    emb_mod = _load_module(
        os.path.join(here, "02_embedding.py"),
        "embedding_mod",
    )
    
    embedder = emb_mod.ParagraphEmbedder(
        batch_size=args.embed_batch_size,
        max_seq_length=args.embed_max_seq_length,
    )
    
    pipeline = AlignmentPipeline(
        zh_dir=args.zh_dir,
        en_dir=args.en_dir,
        tok_dir=args.tok_dir,
        out_dir=args.out_dir,
        vecalign_dir=args.vecalign_dir,
        alignment_max_size=args.alignment_max_size,
        vecalign_extra_args=vecalign_extra_args, # 傳遞參數
    )
    pipeline.set_embedder(embedder)
    pipeline.run()

if __name__ == "__main__":
    main()