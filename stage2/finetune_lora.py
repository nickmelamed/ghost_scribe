"""LoRA fine-tune a small instruction-tuned model on Stage-1's escalated
(ambiguous) training cases only (stage2/data/train.jsonl — see
prepare_finetune_data.py; never the full dataset, per CLAUDE.md).

Model: Qwen2.5-3B-Instruct (open weights, no license gate). Runs on MPS
(Apple Silicon) in bf16; 4-bit bitsandbytes quantization is skipped here
because bitsandbytes doesn't support the MPS backend and this machine
(64GB unified memory) doesn't need it to fit a 3B model — see
requirements.txt, which only installs bitsandbytes on Linux.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "stage2" / "data"
ADAPTER_DIR = ROOT / "stage2" / "adapters" / "qwen2.5-3b-instruct-lora"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# Fixed padding length, not a truncation cap: measured max tokenized
# length (prompt + target) across the escalated train set is 552 tokens
# (see conversation log); 576 gives headroom with zero truncation while
# keeping every batch the same shape (see collate_fn for why fixed shape
# matters on MPS).
MAX_LENGTH = 576

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)


class EscalatedSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = [json.loads(l) for l in path.open()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        messages = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(r["target"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


def collate_fn(batch: list[dict], pad_token_id: int, pad_to: int = MAX_LENGTH) -> dict:
    """Pads every batch to a FIXED length (MAX_LENGTH), not the batch's own
    max — deliberately. Dynamic per-batch padding gives MPS a new tensor
    shape almost every step; over ~200 steps this fragmented the MPS
    allocator badly enough that resident memory grew from ~20GB to ~45GB+
    and step time blew up from ~15s to 200-1000s+ (see git history / the
    conversation this was diagnosed in). Fixed shapes let MPS reuse the
    same buffers step to step.
    """
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad_n = pad_to - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_n)
        labels.append(b["labels"] + [-100] * pad_n)
        attn.append(b["attention_mask"] + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


class MPSCacheClearCallback(TrainerCallback):
    """Extra safety net alongside fixed-length padding: periodically drop
    MPS's cached (but unused) memory blocks so any residual fragmentation
    doesn't accumulate across hundreds of steps."""

    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % self.every_n_steps == 0:
            torch.mps.empty_cache()


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.config.use_cache = False  # required for gradient checkpointing
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    # Needed for gradient checkpointing to work with a frozen base model +
    # LoRA adapters — without this the checkpointed activations have no
    # grad_fn and backward silently fails to save memory (or errors).
    model.enable_input_require_grads()

    train_dataset = EscalatedSFTDataset(DATA_DIR / "train.jsonl", tokenizer)
    print(f"Training on {len(train_dataset)} escalated examples")

    # Batch size kept small with gradient checkpointing on: an earlier full
    # run without checkpointing hit ~75GB resident on this 64GB machine
    # (MPS activation memory for a 3B model does not shed between steps
    # the way CUDA's allocator does) and swapped hard, slowing ~5x. See
    # results/metrics_comparison.md / conversation log for the diagnosis.
    checkpoint_dir = ROOT / "stage2" / "adapters" / "_checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=False,  # MPS Trainer bf16 flag targets CUDA autocast; model is already bf16
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[MPSCacheClearCallback(every_n_steps=10)],
    )
    existing_checkpoints = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
    resume = str(max(existing_checkpoints, key=lambda p: int(p.name.split("-")[1]))) if existing_checkpoints else None
    if resume:
        print(f"Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
