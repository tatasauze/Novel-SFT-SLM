import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

# ======================================================
# English
# ======================================================

DEFAULT_EN_TITLES = [
 'Chapter One',
 'Chapter Two',
 'Chapter Three',
 'Chapter Four',
 'Chapter Five',
 'Chapter Six',
 'Chapter Seven',
 'Chapter Eight',
 'Chapter Nine',
 'Chapter Ten',
 'Chapter Eleven',
 'Chapter Twelve',
 'Chapter Thirteen',
 'Chapter Fourteen',
 'Chapter Fifteen',
 'Chapter Sixteen',
 'Chapter Seventeen',
 'Chapter Eighteen',
 'Chapter Nineteen',
 'Chapter Twenty',
 'Chapter Twenty-One',
 'Chapter Twenty-Two',
 'Chapter Twenty-Three',
 'Chapter Twenty-Four',
 'Chapter Twenty-Five',
 'Chapter Twenty-Six',
 'Chapter Twenty-Seven',
 'Chapter Twenty-Eight',
 'Chapter Twenty-Nine',
 'Chapter Thirty',
 'Chapter Thirty-One',
 'Chapter Thirty-Two',
 'Chapter Thirty-Three',
 'Chapter Thirty-Four',
 'Chapter Thirty-Five',
 'Chapter Thirty-Six',
 'Chapter Thirty-Seven',
 'Chapter Thirty-Eight',
 'Chapter Thirty-Nine',
 'Chapter Forty',
 'Chapter Forty-One',
 'Chapter Forty-Two',
 'Chapter Forty-Three',
 'Chapter Forty-Four',
 'Chapter Forty-Five',
 'Chapter Forty-Six',
 'Chapter Forty-Seven',
 'Chapter Forty-Eight',
 'Chapter Forty-Nine',
 'Chapter Fifty',
 'Chapter Fifty-One',
 'Chapter Fifty-Two',
 'Chapter Fifty-Three',
 'Chapter Fifty-Four',
 'Chapter Fifty-Five',
 'Chapter Fifty-Six',
 'Chapter Fifty-Seven',
 'Chapter Fifty-Eight',
 'Chapter Fifty-Nine',
 'Chapter Sixty',
 'Chapter Sixty-One',
 'Chapter Sixty-Two',
 'Chapter Sixty-Three',
 'Chapter Sixty-Four',
 'Chapter Sixty-Five',
 'Chapter Sixty-Six']
EN_PURE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")

# 段落分隔：空行
EN_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")

# 句末標點，包含常見引號 / 括號 / 右引號
EN_SENTENCE_END_RE = re.compile(r'[.!?…]["\u2019\u201d\)\]\}»]*$')

# 對話引號
# 目前支援：
#   “hello”
#   "hello"
# 如果你的文本使用單引號對話，可以自行加入：
#   ‘hello’
#   'hello'
EN_OPENING_QUOTE_RE = re.compile(r'^[“"]')
EN_CLOSING_QUOTE_END_RE = re.compile(r'["”]\s*$')


def _slugify_en_title(title: str) -> str:
    """
    將章節標題轉成安全的檔案名稱。
    Example:
        Chapter Twenty-One -> chapter_twenty_one
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    return slug or "chapter"


def _dehyphenate_en(text: str) -> str:
    """
    處理英文行尾連字符。

    Example:
        process-
        ing
    becomes:
        processing

    注意：
    這裡使用 [ \\t]* 而不是 \\s*，避免不小心吃掉空行。
    """
    # 移除 soft hyphen
    text = text.replace("\u00ad", "")

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)

    return text


def _normalize_en_paragraph(block: str) -> str:
    """
    將單一英文段落壓成單行。
    段落內的硬換行轉為空格，多重空白壓成單一空格。
    """
    para = re.sub(r"[ \t]*\n[ \t]*", " ", block)
    para = re.sub(r"\s+", " ", para).strip()
    return para


def _is_standalone_quoted_line(line: str) -> bool:
    """
    判斷某一行是否是「獨立的一句對話」。

    例如：
        “Are you going to kill me?”
        "No."
        “Wait.”

    這些通常應該自己成為一個段落。
    """
    line = line.strip()
    if not line:
        return False

    starts_with_quote = bool(EN_OPENING_QUOTE_RE.match(line))
    ends_with_quote = bool(EN_CLOSING_QUOTE_END_RE.search(line))
    ends_with_sentence = bool(EN_SENTENCE_END_RE.search(line))

    return starts_with_quote and ends_with_quote and ends_with_sentence


def _split_en_paragraphs_heuristic(lines: List[str]) -> List[str]:
    """
    當英文文本沒有足夠空行時，用 heuristic 推斷段落。

    適用情境：
    - PDF / OCR 轉出的 TXT
    - 每行都是硬換行
    - 段落之間沒有空行
    - 對話行需要維持獨立段落

    規則：
    1. 如果某一行有縮排，而上一行沒有縮排，視為新段落。
    2. 如果目前這一行以引號開頭，而上一行不是引號開頭，視為新對話段落。
    3. 如果上一行是獨立引號對話，視為上一段結束。
    4. 如果上一行是較短行且以句末標點結束，而目前這一行是正常長度，
       通常代表上一段結束。
    5. 如果上一行以句末標點結束，而目前這一行是極短句，
       目前這一行可能是獨立短段，例如：
           Pure bliss.
    """
    stripped = [line.strip() for line in lines]
    if not stripped:
        return []

    max_len = max(len(s) for s in stripped)

    # 短行 / 極短行門檻
    # 這些值可以依你的文本風格調整
    short_threshold = max(50, int(max_len * 0.75))
    very_short_threshold = max(25, int(max_len * 0.30))

    # ======================================================
    # 如果你希望「所有短句」都強制獨立成段，可以把這裡改成 True。
    # 例如：
    #   Christchurch heat.
    #   Schizophrenic weather.
    # 也會被切成兩段。
    #
    # 預設 False 是為了避免把過短的碎片全部切開。
    # ======================================================
    split_all_short_lines = True

    paragraphs: List[str] = []
    current: List[str] = []

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if i == 0:
            current.append(line)
            continue

        prev_line = stripped[i - 1]
        prev_raw = lines[i - 1]

        prev_ends_sentence = bool(EN_SENTENCE_END_RE.search(prev_line))
        prev_len = len(prev_line)
        curr_len = len(line)

        prev_short = prev_len <= short_threshold
        prev_very_short = prev_len <= very_short_threshold
        curr_very_short = curr_len <= very_short_threshold

        curr_indented = bool(re.match(r"^(?:[ ]{2,}|\t)", raw_line))
        prev_indented = bool(re.match(r"^(?:[ ]{2,}|\t)", prev_raw))

        curr_starts_quote = bool(EN_OPENING_QUOTE_RE.match(line))
        prev_starts_quote = bool(EN_OPENING_QUOTE_RE.match(prev_line))

        prev_standalone_quote = _is_standalone_quoted_line(prev_line)

        should_split = False

        # 1) 行首縮排通常代表新段落
        if curr_indented and not prev_indented:
            should_split = True

        # 2) 目前這一行以引號開頭，而上一行不是引號開頭
        #    通常代表新對話段落。
        if curr_starts_quote and not prev_starts_quote:
            should_split = True

        # 3) 上一行是獨立引號對話
        #    例如：
        #        “Are you going to kill me?”
        #        I shake my head in disbelief.
        #    應該切成兩段。
        if prev_standalone_quote:
            should_split = True

        # 4) 如果你希望所有短句都獨立成段
        if split_all_short_lines and prev_ends_sentence and prev_short:
            should_split = True

        # 5) 上一行是短句，且以句末標點結束；
        #    目前這一行又不是極短句，通常代表新段落開始。
        if (
            prev_ends_sentence
            and prev_short
            and not prev_very_short
            and not curr_very_short
        ):
            should_split = True

        # 6) 上一行以句末標點結束，而目前這一行是極短句，
        #    目前這行很可能是獨立短段。
        if (
            prev_ends_sentence
            and not prev_very_short
            and curr_very_short
        ):
            should_split = True

        if should_split:
            paragraphs.append(" ".join(current))
            current = []

        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return [
        re.sub(r"\s+", " ", p).strip()
        for p in paragraphs
        if p.strip()
    ]


def _extract_en_paragraphs(
    body: str,
    remove_digit_only_lines: bool = True,
    use_heuristic_paragraph_split: bool = True,
) -> List[str]:
    """
    從英文章節內容抽出段落。

    回傳:
        每個段落都是一行。

    分段策略:
        1. 優先使用空行分段。
        2. 對每個空行切出的 block，如果裡面還有多行，
           再使用 heuristic 判斷是否需要進一步分段。
        3. 如果 use_heuristic_paragraph_split=False，
           則只把每個 block 壓成一行。
    """
    raw_lines = body.splitlines()
    processed_lines: List[str] = []

    for line in raw_lines:
        # 移除純數字行，例如頁碼
        if remove_digit_only_lines and EN_PURE_NUMBER_RE.match(line):
            continue

        processed_lines.append(line.rstrip())

    text = "\n".join(processed_lines)

    # --------------------------------------------------
    # 1) 先用空行切 block
    # --------------------------------------------------
    blocks = EN_PARAGRAPH_SPLIT_RE.split(text)
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        return []

    # 如果完全不用 heuristic，每個 block 直接壓成一行
    if not use_heuristic_paragraph_split:
        return [
            _normalize_en_paragraph(block)
            for block in blocks
        ]

    # --------------------------------------------------
    # 2) 對每個 block 再做 heuristic 分段
    # --------------------------------------------------
    paragraphs: List[str] = []

    for block in blocks:
        block_lines = [
            line.rstrip()
            for line in block.splitlines()
            if line.strip()
        ]

        # 如果這個 block 只有一行，直接當成一段
        if len(block_lines) <= 1:
            para = _normalize_en_paragraph(block)
            if para:
                paragraphs.append(para)
            continue

        # 如果這個 block 有多行，嘗試用 heuristic 切段
        paragraphs.extend(
            _split_en_paragraphs_heuristic(block_lines)
        )

    return paragraphs

def _build_en_title_pattern(titles: Sequence[str]):
    clean_titles = [t.strip() for t in titles if t.strip()]
    if not clean_titles:
        return None

    # 先處理較長的標題，避免 Chapter Twenty 被 Chapter Twenty-One 之前誤匹配。
    unique_titles = list(dict.fromkeys(clean_titles))
    escaped_titles = [
        re.escape(t)
        for t in sorted(unique_titles, key=len, reverse=True)
    ]

    # 允許 markdown heading，例如：
    # Chapter One
    # # Chapter One
    # ## Chapter One
    # 也允許結尾冒號，例如：
    # Chapter One:
    pattern = (
        r"^[ \t]*(?:#+[ \t]*)?("
        + "|".join(escaped_titles)
        + r")[ \t:]*\r?$"
    )

    return re.compile(pattern, re.MULTILINE | re.IGNORECASE)


def _canonical_en_title_map(titles: Sequence[str]) -> Dict[str, str]:
    clean_titles = [t.strip() for t in titles if t.strip()]
    return {t.lower(): t for t in clean_titles}


def _split_en_chapters(
    text: str,
    titles: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    根據章節標題拆分英文文本。

    回傳:
        [
            ("Chapter One", chapter_1_body),
            ("Chapter Two", chapter_2_body),
            ...
        ]

    注意：
        如果同一個章節標題在文中重複出現，例如頁面 running header，
        這裡會把相同標題之間的正文合併回同一章，避免被頁眉切碎。
    """
    clean_titles = [t.strip() for t in titles if t.strip()]
    if not clean_titles:
        return []

    title_pattern = _build_en_title_pattern(clean_titles)
    if title_pattern is None:
        return []

    title_map = _canonical_en_title_map(clean_titles)
    matches = list(title_pattern.finditer(text))

    if not matches:
        return []

    chapter_titles: List[str] = []
    chapter_parts: List[List[str]] = []
    last_end: List[int] = []

    for match in matches:
        raw_title = match.group(1).strip()
        title = title_map.get(raw_title.lower(), raw_title)

        # 如果同一章標題再次出現，例如頁眉，
        # 把它視為同一章，保留兩個標題行之間的正文。
        if chapter_titles and chapter_titles[-1] == title:
            chapter_parts[-1].append(text[last_end[-1]:match.start()])
            last_end[-1] = match.end()
        else:
            if chapter_titles:
                chapter_parts[-1].append(text[last_end[-1]:match.start()])

            chapter_titles.append(title)
            chapter_parts.append([])
            last_end.append(match.end())

    if chapter_titles:
        chapter_parts[-1].append(text[last_end[-1]:])

    return [
        (title, "".join(parts))
        for title, parts in zip(chapter_titles, chapter_parts)
    ]


def clean_text_en(
    file_path: str,
    output_dir: str,
    titles: Optional[Sequence[str]] = None,
    include_title: bool = False,
    remove_digit_only_lines: bool = True,
    use_heuristic_paragraph_split: bool = True,
) -> List[str]:
    """
    Functionality:
        1. 先依據 "Chapter" 拆分章節 Chapter One...
        2. 檢查結尾是否有橫槓要將單字連接起來
        3. 每個章節輸出一個檔案
        4. 每個檔案內，一個段落一行

    Arguments:
        file_path: str, path to the input text file
        output_dir: str, output directory for chapter-based files
        titles: list of chapter titles
        include_title: whether to put chapter title as first line in output file
        remove_digit_only_lines: remove standalone numeric lines, e.g. page numbers
        use_heuristic_paragraph_split:
            If True and the chapter has no blank lines, use heuristic paragraph splitting.

    Returns:
        list of output file paths
    """
    if titles is None:
        titles = DEFAULT_EN_TITLES

    text = Path(file_path).read_text(encoding="utf-8-sig")

    # 統一換行符號
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 先處理行尾 hyphen
    text = _dehyphenate_en(text)

    chapters = _split_en_chapters(text, titles)

    output_path = Path(output_dir)

    # 如果使用者不小心傳入 output_en.txt，也轉成 output_en 目錄
    if output_path.suffix:
        output_path = output_path.with_suffix("")

    output_path.mkdir(parents=True, exist_ok=True)

    written_files: List[str] = []

    # 如果完全找不到章節標題，輸出整個清理後文本，方便 debug
    if not chapters:
        paragraphs = _extract_en_paragraphs(
            text,
            remove_digit_only_lines=remove_digit_only_lines,
            use_heuristic_paragraph_split=use_heuristic_paragraph_split,
        )

        out_file = output_path / "full_text.txt"
        content = "\n".join(paragraphs)
        if content:
            content += "\n"

        out_file.write_text(content, encoding="utf-8")
        return [str(out_file)]

    for idx, (title, body) in enumerate(chapters, start=1):
        paragraphs = _extract_en_paragraphs(
            body,
            remove_digit_only_lines=remove_digit_only_lines,
            use_heuristic_paragraph_split=use_heuristic_paragraph_split,
        )

        if include_title:
            paragraphs.insert(0, title)

        file_name = f"{idx}.txt"
        out_file = output_path / file_name

        content = "\n".join(paragraphs)
        if content:
            content += "\n"

        out_file.write_text(content, encoding="utf-8")
        written_files.append(str(out_file))

    return written_files


# ======================================================
# Main
# ======================================================

def main() -> None:

    # --------------------------------------------------
    # EN
    # --------------------------------------------------
    # Since English text files are all TXT files with chapter title,
    # we will use titles to extract context and create separate files
    # for each chapter to be processed.
    
    parser = argparse.ArgumentParser(
        description="依據 PDF 書籤抽取章節文字，輸出為每行一段的 UTF-8 TXT。"
    )

    parser.add_argument(
        "input_file",
        help="輸入 txt 檔案路徑",
    )
    
    parser.add_argument(
        "out_dir",
        help="輸出 txt 資料夾"
    )
    args = parser.parse_args()

    en_input_file = args.input_file
    en_output_dir = args.out_dir
    list_of_titles = DEFAULT_EN_TITLES

    if Path(en_input_file).exists():
        written_files = clean_text_en(
            file_path=en_input_file,
            output_dir=en_output_dir,
            titles=list_of_titles,
            include_title=False,
            remove_digit_only_lines=True,
        )
        print(f"[EN] written {len(written_files)} chapter files into {en_output_dir}/")
    else:
        print(f"[SKIP] EN input not found: {en_input_file}")


if __name__ == "__main__":
    main()