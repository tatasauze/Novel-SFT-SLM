# -*- coding: utf-8 -*-
"""
common_io.py
共用的語料讀取工具，供 /alignment 資料夾內各獨立腳本 import 使用。
不依賴 GPU / 重型套件，僅負責檔案 IO 與段落編號。

輸入資料格式假設：
    ZH: {zh_dir}/output_zh_{i}.txt
    EN: {en_dir}/f{i:02d}_chapter.txt
    每個檔案「一行 = 一個段落」，行序即為段落序。

章節 id 統一格式為 chapter_{i:03d}，段落 id 為 p{line_no}（從 1 開始）。
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


ZH_FILENAME_RE = re.compile(r"^ch(\d+)\.txt$")
EN_FILENAME_RE = re.compile(r"^ch(\d+)\.txt$")


@dataclass
class Chapter:
    chapter_idx: int
    chapter_id: str
    zh_paragraphs: List[str] = field(default_factory=list)
    en_paragraphs: List[str] = field(default_factory=list)
    zh_para_ids: List[str] = field(default_factory=list)
    en_para_ids: List[str] = field(default_factory=list)


class CorpusLoader:
    """負責掃描 output_zh / output_en 目錄，配對章節並讀取段落。"""

    def __init__(self, zh_dir: str = "/output_zh", en_dir: str = "/output_en"):
        self.zh_dir = zh_dir
        self.en_dir = en_dir

    def _scan(self, directory: str, pattern: re.Pattern) -> Dict[int, str]:
        result = {}
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"目錄不存在: {directory}")
        for fname in os.listdir(directory):
            m = pattern.search(fname)
            if m:
                result[int(m.group(1))] = os.path.join(directory, fname)
        return result

    def list_chapter_indices(self) -> List[int]:
        """回傳中英文皆存在對應檔案的章節編號（交集），由小到大排序。"""
        zh_map = self._scan(self.zh_dir, ZH_FILENAME_RE)
        en_map = self._scan(self.en_dir, EN_FILENAME_RE)
        common = sorted(set(zh_map.keys()) & set(en_map.keys()))
        missing_en = sorted(set(zh_map.keys()) - set(en_map.keys()))
        missing_zh = sorted(set(en_map.keys()) - set(zh_map.keys()))
        if missing_en:
            print(f"[CorpusLoader][WARN] 缺少對應英文檔案的章節: {missing_en}")
        if missing_zh:
            print(f"[CorpusLoader][WARN] 缺少對應中文檔案的章節: {missing_zh}")
        return common

    @staticmethod
    def _read_lines(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n").rstrip("\r") for ln in f]
        # 過濾掉純空白行，但保留段落原文（不裁剪內部空白，避免改變格式）
        return [ln for ln in lines if ln.strip() != ""]

    def load_chapter(self, idx: int) -> Chapter:
        zh_path = os.path.join(self.zh_dir, f"ch{idx}.txt")
        en_path = os.path.join(self.en_dir, f"ch{idx}.txt")
        if not os.path.exists(zh_path):
            raise FileNotFoundError(zh_path)
        if not os.path.exists(en_path):
            raise FileNotFoundError(en_path)

        zh_lines = self._read_lines(zh_path)
        en_lines = self._read_lines(en_path)
        
        chapter_id = str(idx)
        return Chapter(
            chapter_idx=idx,
            chapter_id=chapter_id,
            zh_paragraphs=zh_lines,
            en_paragraphs=en_lines,
            zh_para_ids=[f"p{i+1}" for i in range(len(zh_lines))],
            en_para_ids=[f"p{i+1}" for i in range(len(en_lines))],
        )

    def iter_chapters(self):
        for idx in self.list_chapter_indices():
            yield self.load_chapter(idx)


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
