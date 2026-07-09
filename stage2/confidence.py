"""Score Stage 2's confidence in its own verdicts via teacher-forced
two-way scoring, to build a second-tier human review queue.

Stage 2 gets ~99.2% of escalated cases right (Phase 4). Routing every
escalated verdict to a human wastes their time on cases Stage 2already
got right. This computes, for each escalated case, the model's own
probability mass on "Label: AI-generated" vs "Label: Human-written" —
not read off greedy-decode token likelihood, but computed directly by
teacher-forcing both candidate continuations through the model and
comparing their total log-probability. That gives a clean two-way
classification confidence, independent of whatever text the model
happened to generate for the rationale.

IMPORTANT — per CLAUDE.md's explicit no-automated-accusation rule, this
confidence score must only be used to skip human review on
"Human-written" verdicts (auto-clearing takes no action against a
student, so it carries none of the risk the human-confirmation
requirement exists for). Every "AI-generated" verdict still requires
human confirmation regardless of confidence. That policy is applied in
stage2/build_review_queue.py, not here — this module only measures
confidence.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "stage2" / "adapters" / "qwen2.5-3b-instruct-lora"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

CANDIDATE_AI = "Label: AI-generated"
CANDIDATE_HUMAN = "Label: Human-written"
BATCH_SIZE = 16


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(device).eval()
    return model, tokenizer, device


def build_prompts(records: list[dict], tokenizer) -> list[str]:
    prompts = []
    for r in records:
        messages = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


@torch.no_grad()
def score_candidate(model, tokenizer, device, prompts: list[str], candidate_text: str, batch_size: int = BATCH_SIZE) -> list[float]:
    """Total log-probability of `candidate_text` continuing each prompt.

    Left-pads each (prompt + candidate) sequence to the batch's max
    length so the candidate tokens always occupy the same final
    positions regardless of prompt length — makes indexing them uniform
    across a batch of variable-length prompts.
    """
    cand_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
    n_cand = len(cand_ids)
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        full_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] + cand_ids for p in batch]
        max_len = max(len(x) for x in full_ids)

        input_ids, attn = [], []
        for ids in full_ids:
            pad_n = max_len - len(ids)
            input_ids.append([tokenizer.pad_token_id] * pad_n + ids)
            attn.append([0] * pad_n + [1] * len(ids))
        input_ids_t = torch.tensor(input_ids, dtype=torch.long).to(device)
        attn_t = torch.tensor(attn, dtype=torch.long).to(device)

        # Only the last n_cand positions predict the candidate tokens —
        # slice those out BEFORE computing log_softmax over the vocab.
        # The full (unsliced) logits tensor is [batch, seq_len, ~152k
        # vocab]; for a ~600-token sequence that's several GB per batch,
        # and repeatedly materializing it across ~170 batches is what
        # blew MPS memory past 69GB with zero throughput (see commit
        # history / conversation log). Slicing first shrinks the tensor
        # this is computed over by ~150x.
        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
            start = max_len - n_cand - 1
            needed_logits = logits[:, start : start + n_cand, :].float()
            logprobs = torch.log_softmax(needed_logits, dim=-1)

        for b in range(len(batch)):
            total_lp = 0.0
            for k in range(n_cand):
                token_id = input_ids_t[b, max_len - n_cand + k].item()
                total_lp += logprobs[b, k, token_id].item()
            results.append(total_lp)

        del logits, needed_logits, logprobs
        if device == "mps":
            torch.mps.empty_cache()

        print(f"  scored {min(i + batch_size, len(prompts))}/{len(prompts)} ({candidate_text})", flush=True)

    return results


def compute_confidence(model, tokenizer, device, records: list[dict]) -> pd.DataFrame:
    prompts = build_prompts(records, tokenizer)
    lp_ai = score_candidate(model, tokenizer, device, prompts, CANDIDATE_AI)
    lp_human = score_candidate(model, tokenizer, device, prompts, CANDIDATE_HUMAN)

    rows = []
    for r, a, h in zip(records, lp_ai, lp_human):
        m = max(a, h)
        ea, eh = math.exp(a - m), math.exp(h - m)
        p_ai, p_human = ea / (ea + eh), eh / (ea + eh)
        pred = 1 if p_ai > p_human else 0
        rows.append(
            {
                "text": r["text"],
                "generation_source": r["generation_source"],
                "true_label": r.get("label"),
                "confidence_pred_label": pred,
                "confidence": max(p_ai, p_human),
                "p_ai": p_ai,
                "p_human": p_human,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    records = [json.loads(l) for l in (ROOT / "stage2" / "data" / f"{split}.jsonl").open()]
    model, tokenizer, device = load_model()
    df = compute_confidence(model, tokenizer, device, records)
    out_path = ROOT / "data" / "processed" / f"stage2_confidence_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} confidence scores to {out_path}")
