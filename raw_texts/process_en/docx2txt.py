import sys
import re
from docx import Document

# PDF 连字正规化
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

def normalize_ligatures(text: str) -> str:
    for lig, rep in _LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    return text

def para_to_line(para) -> str:
    """把段落压成单行，处理连字与行内换行。"""
    text = para.text
    if not text:
        return ""
    text = normalize_ligatures(text)
    # 行内换行（<w:br/>）转成空格
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    # 压缩多重空白
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_paragraphs(docx_path: str) -> list:
    doc = Document(docx_path)
    paragraphs = []

    # 正文段落
    for para in doc.paragraphs:
        line = para_to_line(para)
        if line:
            paragraphs.append(line)

    # 如果诊断发现表格里有正文，取消下面的注释
    # for table in doc.tables:
    #     for row in table.rows:
    #         for cell in row.cells:
    #             for para in cell.paragraphs:
    #                 line = para_to_line(para)
    #                 if line:
    #                     paragraphs.append(line)

    return paragraphs

def main():
    if len(sys.argv) != 3:
        print("Usage: python docx2txt.py input.docx output.txt")
        sys.exit(1)

    docx_path, txt_path = sys.argv[1], sys.argv[2]
    paragraphs = extract_paragraphs(docx_path)

    # 关键：每段一行，段与段之间用空行分隔
    with open(txt_path, "w", encoding="utf-8") as f:
        content = "\n\n".join(paragraphs)
        # 忽略大小寫 (re.IGNORECASE)，並容許 com 前後有任意半形/全形空白
        content = re.sub(r"OceanofPDF\s*\.\s*com", "", content, flags=re.IGNORECASE)
        f.write(content)
        f.write("\n")

    print(f"[OK] extracted {len(paragraphs)} paragraphs -> {txt_path}")

if __name__ == "__main__":
    main()