import glob
import os


def check_line_conditions(line: str) -> list[str]:
    """檢視單行文字是否滿足條件，傳回不滿足的原因清單。"""
    errors = []

    # 條件 1：沒有半形 ( ) 或全形 ( ) 空格
    
    if " " in line or " " in line:
        errors.append("包含空格 (半形或全形)")

    
    # 條件 2：「 與 」 數量必須匹配
    open_quote_count = line.count("「")
    close_quote_count = line.count("」")
    if open_quote_count != close_quote_count:
        errors.append(
            f"引號不匹配 (「: {open_quote_count} 個, 」: {close_quote_count} 個)"
        )

    # 條件 3：是否存在半形 | 或全形 ｜ (依需求：若存在則標記異常)
    
    if "|" in line or "｜" in line:
        errors.append("包含管道符號 (| 或 ｜)")
    
    return errors


def inspect_txt_files(folder_path: str):
    """迭代資料夾中的所有 .txt 檔案並檢查內容。"""
    if not os.path.exists(folder_path):
        print(f"錯誤：找不到資料夾路徑 '{folder_path}'")
        return

    # 搜尋資料夾下的所有 .txt 檔
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not txt_files:
        print(f"在 '{folder_path}' 中未找到任何 .txt 檔案。")
        return

    total_issues = 0

    for file_path in txt_files:
        file_name = os.path.basename(file_path)

        try:
            # 使用 utf-8 讀取，遇到編碼問題自動替換避免程式中斷
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    # 移除行末換行符號後再進行檢查
                    clean_line = line.rstrip("\r\n")

                    errors = check_line_conditions(clean_line)

                    if errors:
                        total_issues += 1
                        error_msg = "、".join(errors)
                        print(
                            f"【檔名: {file_name}】 [第 {line_num} 行] - 未滿足條件: {error_msg}"
                        )
                        print(f"   內容: {clean_line}\n")

        except Exception as e:
            print(f"無法讀取檔案 {file_name}: {e}")

    if total_issues == 0:
        print("檢測完成！所有 .txt 檔案均符合格式要求。")
    else:
        print(f"檢測完成，共發現 {total_issues} 處不符合條件的行數。")


# ------------------ 使用範例 ------------------
if __name__ == "__main__":
    # 請替換為你要檢查的資料夾路徑
    target_folder = "./zh"

    inspect_txt_files(target_folder)