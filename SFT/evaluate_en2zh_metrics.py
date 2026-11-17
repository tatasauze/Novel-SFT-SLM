#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
英中小說翻譯 LoRA SFT 訓練後評估腳本。

評估指標：
- BLEU: sacrebleu, tokenize="zh"
- chrF++: sacrebleu corpus_chrf, word_order=2
- COMET: Unbabel/wmt22-comet-da
- BERTScore: lang="zh"

使用範例：

python evaluate_en2zh_metrics.py \
    --csv ./aligned_data.csv \
    --adapter_path ./qwen3-4b-en2zh-sft/final_lora \
    --split test \
    --max_seq_length 2048 \
    --max_new_tokens 256 \
    --max_samples -1 \
    --metrics bleu chrf comet bertscore

T4 如果擔心 OOM，可以先這樣跑：

python evaluate_en2zh_metrics.py \
    --csv ./aligned_data.csv \
    --adapter_path ./qwen3-4b-en2zh-sft/final_lora \
    --split test \
    --max_seq_length 2048 \
    --max_new_tokens 128 \
    --max_samples 200 \
    --metrics bleu chrf

重要：
- 請盡量使用與訓練時相同的 max_seq_length、seed、train_ratio/val_ratio/test_ratio。
- 建議訓練結束後另開一個 process 執行本腳本，不要在訓練的同一個 process 內接續跑，避免 VRAM 峰值過高。
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import gc
import json
import time
from typing import List, Set

import pandas as pd
import torch

# 從你的訓練腳本導入相同的前處理、切分與模板
# 請確認此腳本與 train_en2zh_sft.py 在同一目錄，或 PYTHONPATH 可找到它
from train_en2zh_sft import (
    TrainingConfig,
    preprocess_dataframe,
    compute_token_lengths,
    split_by_chapter,
    PROMPT_TEMPLATE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate en2zh LoRA SFT model with BLEU, chrF++, COMET, BERTScore."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="./aligned_data.csv",
        help="訓練/評估使用的 CSV，需包含 en_text, zh_text，最好包含 chapter_id。",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./qwen3-4b-en2zh-sft/final_lora",
        help="train_en2zh_sft.py 儲存出來的 LoRA adapter 目錄。",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
        help="Base model。通常與訓練腳本一致。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="要評估哪一個 split。",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="請盡量與訓練時相同。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成時最多產生多少 tokens。T4 建議先用 128 或 256。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最多評估幾筆。-1 表示全部。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="請與訓練時相同。",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.80,
        help="請與訓練時相同。",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="請與訓練時相同。",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="請與訓練時相同。",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "chrf", "comet", "bertscore"],
        choices=["bleu", "chrf", "comet", "bertscore", "all"],
        help="要計算哪些指標。",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help="COMET 模型。也可改用 Unbabel/wmt22-cometkiwi-da。",
    )
    parser.add_argument(
        "--comet_batch_size",
        type=int,
        default=2,
        help="COMET batch size。T4 建議 1 或 2。",
    )
    parser.add_argument(
        "--bertscore_lang",
        type=str,
        default="zh",
        help="BERTScore 語言。中文用 zh。",
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=4,
        help="BERTScore batch size。T4 可降到 1 或 2。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="評估結果輸出目錄。",
    )
    parser.add_argument(
        "--no_filter_outlier_alignment",
        action="store_true",
        help="不要過濾 outlier_alignment。預設會與訓練腳本一致進行過濾。",
    )
    parser.add_argument(
        "--no_filter_token_ratio",
        action="store_true",
        help="不要過濾極端 token ratio。預設會與訓練腳本一致進行過濾。",
    )

    return parser.parse_args()


def expand_metrics(metrics: List[str]) -> Set[str]:
    metrics_set = set(metrics)
    if "all" in metrics_set:
        return {"bleu", "chrf", "comet", "bertscore"}
    return metrics_set


def is_adapter_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False

    candidate_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    return any(
        os.path.exists(os.path.join(path, filename))
        for filename in candidate_files
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    """
    載入 base model + LoRA adapter。

    優先嘗試直接用 FastLanguageModel.from_pretrained(adapter_path)。
    若失敗，則先載入 base model，再載入 adapter。
    """
    from unsloth import FastLanguageModel

    adapter_path = (args.adapter_path or "").strip()

    if adapter_path.lower() in {"", "none", "null", "base"}:
        adapter_path = ""

    if is_adapter_dir(adapter_path):
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            print(f"[*] 已直接載入 adapter: {adapter_path}")
        except Exception as e_direct:
            print(f"[!] 直接載入 adapter 失敗：{e_direct}")
            print("[*] 改為先載入 base model，再載入 adapter。")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            try:
                model.load_adapter(adapter_path)
                print(f"[*] 使用 model.load_adapter() 載入: {adapter_path}")
            except Exception as e_load_adapter:
                print(f"[!] model.load_adapter() 失敗：{e_load_adapter}")
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(
                        model,
                        adapter_path,
                    )
                    print(f"[*] 使用 PeftModel.from_pretrained() 載入: {adapter_path}")
                except Exception as e_peft:
                    raise RuntimeError(
                        f"無法載入 adapter：{adapter_path}。\n"
                        f"load_adapter 錯誤：{e_load_adapter}\n"
                        f"PeftModel 錯誤：{e_peft}"
                    )
    else:
        print("[!] 找不到 adapter 目錄，將只載入 base model。")
        print("    若你要評估微調模型，請確認 --adapter_path 是否正確。")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

    # 若 adapter 目錄內有 tokenizer，優先使用 adapter 目錄的 tokenizer
    if is_adapter_dir(adapter_path) and os.path.exists(
        os.path.join(adapter_path, "tokenizer_config.json")
    ):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            print("[*] Tokenizer 已從 adapter 目錄載入。")
        except Exception as e_tokenizer:
            print(f"[!] 無法從 adapter 目錄載入 tokenizer：{e_tokenizer}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 生成時通常使用 left padding 比較安全；單筆生成影響較小
    tokenizer.padding_side = "left"

    try:
        FastLanguageModel.for_inference(model)
        print("[*] 已呼叫 FastLanguageModel.for_inference(model)。")
    except Exception as e_inference:
        print(f"[!] FastLanguageModel.for_inference 失敗，改用 model.eval(): {e_inference}")
        model.eval()

    return model, tokenizer


def clean_generated_text(text: str, tokenizer) -> str:
    """
    清除生成結果中的模板符號與 EOS。
    """
    for marker in ["### English", "### Chinese"]:
        text = text.replace(marker, "")

    if tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, "")

    text = text.strip()

    return text


def generate_hypotheses(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
):
    """
    逐筆生成翻譯結果。

    為了避免 T4 OOM，預設逐筆生成，不做大 batch。
    """
    sources = eval_df["en_text"].astype(str).tolist()
    references = eval_df["zh_text"].astype(str).tolist()
    hypotheses = []

    device = next(model.parameters()).device

    # 保留一點空間給 prompt，避免 prompt + new_tokens 超過 max_seq_length
    max_prompt_tokens = max(
        16,
        args.max_seq_length - args.max_new_tokens - 8,
    )

    print("\n" + "=" * 60)
    print("開始生成翻譯")
    print("=" * 60)
    print(f"  樣本數: {len(sources)}")
    print(f"  max_seq_length: {args.max_seq_length}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  max_prompt_tokens: {max_prompt_tokens}")

    try:
        from tqdm import tqdm

        iterator = tqdm(range(len(sources)), desc="Generating", ncols=90)
        has_tqdm = True
    except Exception:
        iterator = range(len(sources))
        has_tqdm = False

    for i in iterator:
        en_text = sources[i]

        prompt_str = PROMPT_TEMPLATE.format(en_text=en_text)

        input_ids = tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
        )

        if len(input_ids) > max_prompt_tokens:
            input_ids = input_ids[:max_prompt_tokens]

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_tensor,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        generated_ids = output_ids[0][input_tensor.shape[1]:]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        generated_text = clean_generated_text(generated_text, tokenizer)

        if not generated_text:
            generated_text = " "

        hypotheses.append(generated_text)

        del input_tensor, output_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not has_tqdm and (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{len(sources)}")

    return sources, references, hypotheses


def save_translations_jsonl(
    records: List[dict],
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"[*] 翻譯結果已儲存至: {output_path}")


def compute_sacrebleu_and_chrf(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """
    計算 BLEU 與 chrF++。
    chrF++ 使用 word_order=2。
    """
    import sacrebleu

    # BLEU，中文建議使用 tokenize="zh"
    try:
        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            tokenize="zh",
        )
    except Exception as e_bleu_zh:
        print(f"[!] sacrebleu tokenize='zh' 失敗，改用 tokenize='none': {e_bleu_zh}")
        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            tokenize="none",
        )

    # chrF++
    try:
        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            word_order=2,
        )
    except Exception as e_chrf:
        print(f"[!] sacrebleu corpus_chrf(word_order=2) 失敗，改用預設: {e_chrf}")
        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
        )

    return {
        "bleu": {
            "score": float(bleu.score),
            "precisions": [float(p) for p in getattr(bleu, "precisions", [])],
            "bp": float(getattr(bleu, "bp", 0.0)),
            "sys_len": int(getattr(bleu, "sys_len", 0)),
            "ref_len": int(getattr(bleu, "ref_len", 0)),
        },
        "chrf_plus_plus": {
            "score": float(chrf.score),
            "precision": float(getattr(chrf, "precision", 0.0)),
            "recall": float(getattr(chrf, "recall", 0.0)),
        },
    }


def compute_comet_metric(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    comet_model_name: str,
    batch_size: int,
) -> dict:
    """
    計算 COMET。

    使用格式：
    [
        {"src": source, "mt": hypothesis, "ref": reference},
        ...
    ]
    """
    from comet import download_model, load_from_checkpoint

    print(f"[*] 下載 / 載入 COMET 模型: {comet_model_name}")
    model_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(model_path)

    data = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        data.append(
            {
                "src": src,
                "mt": hyp,
                "ref": ref,
            }
        )

    attempts = []

    if torch.cuda.is_available():
        attempts.append({"accelerator": "gpu", "devices": 1})
        attempts.append({"gpus": 1})

    attempts.append({})

    output = None
    last_error = None

    for kwargs in attempts:
        try:
            output = comet_model.predict(
                data,
                batch_size=batch_size,
                **kwargs,
            )
            break
        except Exception as e:
            last_error = e

    if output is None:
        raise RuntimeError(f"COMET predict 失敗，最後錯誤：{last_error}")

    system_score = None
    scores = []

    if hasattr(output, "system_score"):
        system_score = float(output.system_score)
    elif isinstance(output, dict) and "system_score" in output:
        system_score = float(output["system_score"])

    if hasattr(output, "scores"):
        scores = [float(x) for x in output.scores]
    elif isinstance(output, dict) and "scores" in output:
        scores = [float(x) for x in output["scores"]]

    mean_score = (
        float(sum(scores) / len(scores))
        if scores
        else (system_score if system_score is not None else 0.0)
    )

    del comet_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": comet_model_name,
        "system_score": system_score,
        "mean_score": mean_score,
        "num_scores": len(scores),
    }


def compute_bertscore_metric(
    hypotheses: List[str],
    references: List[str],
    lang: str,
    batch_size: int,
) -> dict:
    """
    計算 BERTScore。
    """
    from bert_score import score as bert_score

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        P, R, F1 = bert_score(
            hypotheses,
            references,
            lang=lang,
            device=device,
            batch_size=batch_size,
            verbose=True,
        )
    except Exception as e:
        error_message = str(e).lower()

        if torch.cuda.is_available() and "out of memory" in error_message:
            print("[!] BERTScore GPU OOM，嘗試改用 CPU。")
            gc.collect()
            torch.cuda.empty_cache()

            P, R, F1 = bert_score(
                hypotheses,
                references,
                lang=lang,
                device="cpu",
                batch_size=max(1, batch_size // 2),
                verbose=True,
            )
        else:
            raise e

    return {
        "lang": lang,
        "f1_mean": float(F1.mean().item()),
        "precision_mean": float(P.mean().item()),
        "recall_mean": float(R.mean().item()),
    }


def main():
    args = parse_args()

    # 避免 max_new_tokens 太大導致 prompt 空間不足
    max_allowed_new_tokens = max(16, args.max_seq_length - 64)
    if args.max_new_tokens > max_allowed_new_tokens:
        print(
            f"[!] max_new_tokens={args.max_new_tokens} 對 max_seq_length={args.max_seq_length} 太大，"
            f"自動調整為 {max_allowed_new_tokens}。"
        )
        args.max_new_tokens = max_allowed_new_tokens

    os.makedirs(args.output_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics_path = os.path.join(args.output_dir, f"metrics_{run_id}.json")
    translations_path = os.path.join(args.output_dir, f"translations_{args.split}_{run_id}.jsonl")

    print("=" * 60)
    print("英中小說翻譯 LoRA SFT 評估")
    print("=" * 60)
    print(f"  CSV: {args.csv}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Split: {args.split}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Metrics: {args.metrics}")

    # ------------------------------------------------------
    # 1. 載入模型與 tokenizer
    # ------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args)

    # ------------------------------------------------------
    # 2. 建立與訓練腳本一致的前處理與切分設定
    # ------------------------------------------------------
    cfg = TrainingConfig(
        csv_path=args.csv,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    cfg.filter_outlier_alignment = not args.no_filter_outlier_alignment
    cfg.filter_by_token_ratio = not args.no_filter_token_ratio

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"找不到 CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"[*] 從 {args.csv} 載入 {len(df)} 筆資料。")

    df_clean = preprocess_dataframe(df, cfg)
    df_clean = compute_token_lengths(df_clean, tokenizer, cfg)

    train_df, val_df, test_df = split_by_chapter(df_clean, cfg)

    if args.split == "train":
        eval_df = train_df.copy()
    elif args.split == "val":
        eval_df = val_df.copy()
    elif args.split == "test":
        eval_df = test_df.copy()
    else:
        raise ValueError(f"不支援的 split: {args.split}")

    eval_df = eval_df.reset_index(drop=True)

    if args.max_samples > 0:
        eval_df = eval_df.iloc[: args.max_samples].reset_index(drop=True)

    if len(eval_df) == 0:
        raise RuntimeError("評估資料集為空。請檢查 CSV、split、max_samples 或過濾條件。")

    print(f"[*] 使用 {args.split} split 進行評估，共 {len(eval_df)} 筆。")

    # ------------------------------------------------------
    # 3. 生成翻譯
    # ------------------------------------------------------
    sources, references, hypotheses = generate_hypotheses(
        model=model,
        tokenizer=tokenizer,
        eval_df=eval_df,
        args=args,
    )

    # ------------------------------------------------------
    # 4. 先儲存翻譯結果
    # ------------------------------------------------------
    records = []

    chapter_ids = (
        eval_df["chapter_id"].tolist()
        if "chapter_id" in eval_df.columns
        else [None] * len(eval_df)
    )

    for i, (src, ref, hyp, chapter_id) in enumerate(
        zip(sources, references, hypotheses, chapter_ids)
    ):
        records.append(
            {
                "id": i,
                "split": args.split,
                "chapter_id": chapter_id,
                "source": src,
                "reference": ref,
                "hypothesis": hyp,
            }
        )

    save_translations_jsonl(records, translations_path)

    # ------------------------------------------------------
    # 5. 刪除大型翻譯模型，釋放 VRAM，再計算其他指標
    # ------------------------------------------------------
    print("\n[*] 釋放翻譯模型以保留足夠 VRAM 給 COMET / BERTScore。")
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------
    # 6. 計算指標
    # ------------------------------------------------------
    metrics_to_run = expand_metrics(args.metrics)
    all_metrics = {}

    # BLEU / chrF++
    if "bleu" in metrics_to_run or "chrf" in metrics_to_run:
        try:
            sacre_metrics = compute_sacrebleu_and_chrf(
                hypotheses=hypotheses,
                references=references,
            )
            if "bleu" in metrics_to_run:
                all_metrics["bleu"] = sacre_metrics.get("bleu", {})
            if "chrf" in metrics_to_run:
                all_metrics["chrf_plus_plus"] = sacre_metrics.get("chrf_plus_plus", {})
        except Exception as e:
            print(f"[!] sacrebleu 計算失敗：{e}")

    # COMET
    if "comet" in metrics_to_run:
        try:
            comet_result = compute_comet_metric(
                sources=sources,
                hypotheses=hypotheses,
                references=references,
                comet_model_name=args.comet_model,
                batch_size=args.comet_batch_size,
            )
            all_metrics["comet"] = comet_result
        except Exception as e:
            print(f"[!] COMET 計算失敗：{e}")
            print("    若未安裝，可執行：pip install unbabel-comet")

    # BERTScore
    if "bertscore" in metrics_to_run:
        try:
            bertscore_result = compute_bertscore_metric(
                hypotheses=hypotheses,
                references=references,
                lang=args.bertscore_lang,
                batch_size=args.bertscore_batch_size,
            )
            all_metrics["bertscore"] = bertscore_result
        except Exception as e:
            print(f"[!] BERTScore 計算失敗：{e}")
            print("    若未安裝，可執行：pip install bert-score")

    # ------------------------------------------------------
    # 7. 儲存 metrics.json
    # ------------------------------------------------------
    result = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "num_samples": len(hypotheses),
        "split": args.split,
        "metrics": all_metrics,
        "translations_path": translations_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("評估完成")
    print("=" * 60)
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Translations: {translations_path}")

    print("\n[指標摘要]")

    if "bleu" in all_metrics:
        print(f"  BLEU: {all_metrics['bleu'].get('score', None)}")

    if "chrf_plus_plus" in all_metrics:
        print(f"  chrF++: {all_metrics['chrf_plus_plus'].get('score', None)}")

    if "comet" in all_metrics:
        print(f"  COMET system_score: {all_metrics['comet'].get('system_score', None)}")
        print(f"  COMET mean_score: {all_metrics['comet'].get('mean_score', None)}")

    if "bertscore" in all_metrics:
        print(f"  BERTScore F1: {all_metrics['bertscore'].get('f1_mean', None)}")


if __name__ == "__main__":
    main()