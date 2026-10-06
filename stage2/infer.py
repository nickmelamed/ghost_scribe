"""Run the LoRA fine-tuned model on escalated cases: label + rationale.

Usage:
    python -m stage2.infer --split test               # full escalated test set
    python -m stage2.infer --split test --n_sample 20  # quick sample, for inspection
    python -m stage2.infer --split esl_eval            # escalated ESL fairness subset

Output: data/processed/stage2_predictions_{split}.csv, joined back to the
full pipeline by essay text, which is unique across the dataset, since prepare_finetune_data.py's escalated-only jsonl files
don't retain the original Stage-1 dataframe index.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "stage2" / "data"
ADAPTER_DIR = ROOT / "stage2" / "adapters" / "qwen2.5-3b-instruct-lora"
OUT_DIR = ROOT / "data" / "processed"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 64

LABEL_RE = re.compile(r"Label:\s*(AI-generated|Human-written)", re.IGNORECASE)
RATIONALE_RE = re.compile(r"Rationale:\s*(.*)", re.IGNORECASE | re.DOTALL)


def load_model(adapter_dir: Path = ADAPTER_DIR):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.to(device).eval()
    return model, tokenizer, device


def parse_output(text: str) -> tuple[int | None, str]:
    label_match = LABEL_RE.search(text)
    rationale_match = RATIONALE_RE.search(text)
    label = None
    if label_match:
        label = 1 if label_match.group(1).lower() == "ai-generated" else 0
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    return label, rationale


@torch.no_grad()
def generate_batch(model, tokenizer, device, records: list[dict], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=896).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j in range(len(batch)):
            new_tokens = gen[j][inputs["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(records))}/{len(records)}")
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="name of a JSONL file in the data folder, such as test")
    parser.add_argument("--n_sample", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    records = [json.loads(line) for line in (args.data_dir / f"{args.split}.jsonl").open()]
    if args.n_sample:
        records = records[: args.n_sample]

    print(f"Running inference on {len(records)} escalated {args.split} rows (batch_size={args.batch_size})")
    model, tokenizer, device = load_model(args.adapter_dir)
    print(f"Model + adapter loaded on {device}")

    raw_outputs = generate_batch(model, tokenizer, device, records, args.batch_size, args.max_new_tokens)

    rows = []
    for r, raw in zip(records, raw_outputs):
        pred_label, rationale = parse_output(raw)
        rows.append(
            {
                "text": r["text"],
                "true_label": r["label"],
                "generation_source": r["generation_source"],
                "stage2_pred_label": pred_label,
                "stage2_rationale": rationale,
                "stage2_raw": raw,
            }
        )

    out_df = pd.DataFrame(rows)
    n_unparsed = out_df["stage2_pred_label"].isnull().sum()
    if n_unparsed:
        print(f"WARNING: {n_unparsed}/{len(out_df)} outputs did not match the expected Label: format")

    suffix = f"_sample{args.n_sample}" if args.n_sample else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"stage2_predictions_{args.split}{suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if args.n_sample:
        for _, row in out_df.head(5).iterrows():
            print(f"\ntrue={row['true_label']} pred={row['stage2_pred_label']}")
            print(f"  {row['stage2_raw'][:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
