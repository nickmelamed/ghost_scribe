"""Independent LLM review pass over Stage-1-escalated essays, as a
point of comparison alongside real human review (hitl/review_app.py).

Uses the BASE Qwen2.5-3B-Instruct model — no LoRA adapter — specifically
so it is not the same model being asked to grade its own homework: Stage
2 is this same base model *plus* a LoRA adapter fine-tuned on escalated
cases, so it would be a close-to-circular check to use it here too. The
base model sees the essay and Stage 2's suggested verdict + rationale
and is asked to independently agree, overturn, and tag failure modes —
the same task the human reviewer app gives a teacher — but it is an LLM
opinion, not a human one, and should never be reported as the latter.

Output: hitl/llm_reviews.jsonl, schema-compatible with hitl/reviews.jsonl
(same fields plus reviewer_type="llm_base_model" and the model id) but
written to a SEPARATE file so it never collides with, or gets skipped
by, a real human review pass over the same escalated pool.

Run with:
    python -m hitl.llm_review
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hitl.review_app import FAILURE_TAGS

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "data" / "processed" / "stage2_predictions_val.csv"
OUT_PATH = ROOT / "hitl" / "llm_reviews.jsonl"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVIEWER_TYPE = "llm_base_model"
MAX_ESSAY_CHARS = 1500
MAX_NEW_TOKENS = 160
BATCH_SIZE = 8

# Prompt-engineering history (see conversation/commit log for full
# numbers), kept here so this isn't silently re-litigated later:
#   v1 (no rubric, "academic-integrity triage" framing): 61.7% accuracy,
#     but degenerate — defaulted to "Human-written" on 96.6% of cases.
#   v2 (this version — neutral framing, explicit stylometric rubric,
#     debiasing instruction, reasoning-before-answer): ~64% accuracy,
#     9% unparsed, and — importantly — non-degenerate, balanced
#     predictions. The best-calibrated version tested.
#   v3 (v2 + two worked few-shot examples): looked better on the surface
#     (75.8%) but that was fallback contamination from a truncation bug
#     (52% unparsed, silently falling back to Stage 2's own ~99%-accurate
#     label). Once fixed and fully isolated from fallback rows, true
#     accuracy was 56.5% — barely above chance, and biased toward
#     over-predicting "AI-generated" — i.e. few-shot examples made a
#     non-fine-tuned model *worse*, not better. Reverted.
# Conclusion: an off-the-shelf 3B instruct model, no matter how it's
# prompted, tops out well below Stage 2's 99.2% fine-tuned accuracy on
# this task. Keep this as a documented comparison baseline — do not feed
# its output into Phase 6 retraining.
SYSTEM_PROMPT = (
    "You are analyzing a student essay for stylistic evidence of AI "
    "generation, as a second opinion alongside a fine-tuned model's "
    "suggestion. This is text analysis, not a disciplinary judgment — "
    "base your answer only on observable stylistic evidence in the essay.\n\n"
    "Consider cues like:\n"
    "- Sentence-length variability: human writing tends to vary sentence "
    "length more; AI-generated text is often more uniform.\n"
    "- Vocabulary and word length: AI-generated essays often use longer, "
    "more uniformly \"sophisticated\" words; human writing has more "
    "variation, including simpler or informal word choices.\n"
    "- Punctuation patterns: unusual comma frequency or highly regular "
    "punctuation can indicate AI generation.\n"
    "- Repetition and phrasing: AI text sometimes reuses phrasing or "
    "sentence structure; human writing has more idiosyncratic repetition.\n"
    "- Readability: extremely smooth, uniformly graded-level prose "
    "throughout the whole essay can indicate AI generation.\n\n"
    "About half of the essays you will review are actually AI-generated — "
    "do not default to \"Human-written\" without genuinely weighing the "
    "evidence above. Use the other model's suggestion as one input, not "
    "the final answer — form your own view from the essay itself.\n\n"
    "Also decide whether a failure mode applies, most importantly whether "
    "the essay may be getting flagged due to non-native English writing "
    "style rather than actual AI generation. This is a decision-support "
    "suggestion only; a human teacher must still confirm any final "
    "determination.\n\n"
    "Respond in exactly this format — Determination must be exactly one "
    "of the two words shown, never a hedge or a failure tag:\n"
    "Analysis: <1-2 sentences on the specific stylistic cues you observed>\n"
    "Determination: <Human-written or AI-generated>\n"
    "Failure tag: <None, or 'possible false flag: non-native writing "
    "style', or 'Stage 2 rationale doesn't match the essay', or 'Other / "
    "unsure'>"
)

ANALYSIS_RE = re.compile(r"Analysis:\s*(.*?)(?:\n *Determination:|\Z)", re.DOTALL)
# Substring match, not exact: the model sometimes hedges ("Possible
# AI-generated") — still a usable answer as long as one of the two
# labels appears in the Determination line.
DETERMINATION_RE = re.compile(r"Determination:\s*(.*?)(?:\n|$)")
TAG_RE = re.compile(r"Failure tag:\s*(.*?)(?:\n|$)")


def build_user_message(text: str, stage2_label: str, stage2_rationale: str) -> str:
    truncated = text[:MAX_ESSAY_CHARS]
    suffix = " [...]" if len(text) > MAX_ESSAY_CHARS else ""
    return (
        f"Essay:\n{truncated}{suffix}\n\n"
        f"Other model's suggestion: {stage2_label} — {stage2_rationale}"
    )


def parse_output(raw: str) -> tuple[int | None, str, str]:
    det_match = DETERMINATION_RE.search(raw)
    label = None
    if det_match:
        det_text = det_match.group(1).lower()
        has_ai = "ai-generated" in det_text
        has_human = "human-written" in det_text
        if has_ai and not has_human:
            label = 1
        elif has_human and not has_ai:
            label = 0
        # both or neither present (e.g. a failure tag leaked into this
        # field) -> label stays None, falls back to Stage 2 downstream

    tag_match = TAG_RE.search(raw)
    tag = tag_match.group(1).strip().strip("'\"") if tag_match else ""
    if tag not in FAILURE_TAGS:
        tag = "Other / unsure" if tag else "None"

    analysis_match = ANALYSIS_RE.search(raw)
    analysis = analysis_match.group(1).strip() if analysis_match else ""

    return label, tag, analysis


@torch.no_grad()
def generate_batch(model, tokenizer, device, prompts: list[str]) -> list[str]:
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=896).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j in range(len(batch)):
            new_tokens = gen[j][inputs["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        print(f"  reviewed {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}")
    return outputs


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Base model only — deliberately no LoRA adapter, see module docstring.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()

    df = pd.read_csv(PREDICTIONS_PATH).drop(columns=["true_label"])
    print(f"Running independent LLM review on {len(df)} escalated essays (base model, no adapter)")

    prompts = []
    for _, row in df.iterrows():
        stage2_label = "AI-generated" if row["stage2_pred_label"] == 1 else "Human-written"
        user_msg = build_user_message(row["text"], stage2_label, row["stage2_rationale"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    raw_outputs = generate_batch(model, tokenizer, device, prompts)

    now = datetime.now(timezone.utc).isoformat()
    n_unparsed = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row, raw in zip(df.itertuples(), raw_outputs):
            reviewer_label, tag, notes = parse_output(raw)
            if reviewer_label is None:
                n_unparsed += 1
                reviewer_label = row.stage2_pred_label  # documented fallback, not silent
            record = {
                "text": row.text,
                "generation_source": row.generation_source,
                "stage2_pred_label": int(row.stage2_pred_label),
                "stage2_rationale": row.stage2_rationale,
                "reviewer_label": int(reviewer_label),
                "reviewer_agreed": int(reviewer_label) == int(row.stage2_pred_label),
                "failure_tag": tag,
                "notes": notes,
                "reviewer_type": REVIEWER_TYPE,
                "reviewer_model": MODEL_ID + " (base, no adapter)",
                "reviewed_at": now,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(df)} LLM reviews to {OUT_PATH} ({n_unparsed} unparsed, fell back to Stage 2's label)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
