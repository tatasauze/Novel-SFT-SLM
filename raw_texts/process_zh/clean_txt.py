import os
from pathlib import Path


def clean_txt_files(input_dir: str, output_dir: str):
    """讀取 input_dir 中的所有 .txt 檔案，執行清理後輸出至 output_dir。

    :param input_dir: 原始 txt 檔案的資料夾路徑
    :param output_dir: 清理後檔案的輸出資料夾路徑
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # 確保輸出目錄存在，若不存在則自動建立
    out_path.mkdir(parents=True, exist_ok=True)

    # 搜尋 input_dir 底下所有的 .txt 檔案
    txt_files = list(in_path.glob("*.txt"))

    if not txt_files:
        print(f"在 '{input_dir}' 中未找到任何 .txt 檔案。")
        return

    print(f"找到 {len(txt_files)} 個 .txt 檔案，開始清理...\n")

    for file_path in txt_files:
        cleaned_lines = []

        # 讀取檔案內容
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # 1. 檢查是否包含 "###"
                if "###" in line:
                    continue

                # 2. 檢查整行去除前後空白/換行後是否為空行
                if not line.strip():
                    continue

                # 符合條件則保留該行
                cleaned_lines.append(line)

        # 寫入至 output_dir 中的同名檔案
        output_file_path = out_path / file_path.name
        with open(output_file_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(cleaned_lines)

        print(
            f"[OK] {file_path.name} -> 保留 {len(cleaned_lines)} 行，已寫入 {output_file_path}"
        )

    print("\n所有檔案清理完成！")


# ------------------ 使用範例 ------------------
if __name__ == "__main__":
    input_directory = r"C:\Users\User\OneDrive\桌面\novel SFT\raw_texts\zh"  # 原始檔案資料夾
    output_directory = "./cleaned_txt_files"  # 清理後輸出資料夾

    clean_txt_files(input_directory, output_directory)