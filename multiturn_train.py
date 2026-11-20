import torch
assert torch.cuda.is_available(), "Enable GPU runtime"

# =============================================================
# MULTI-TURN SFT TRAINER — Stage 2b
#
# ⚠️ PREREQUISITES — run these two scripts first:
#   1. python SLM_finetune_v2.py          (Stage 1 — single-turn SFT)
#   2. python generate_multiturn_dataset.py (Stage 2a — expand dataset)
#
# This script loads the Stage 1 adapter and trains it further on
# the multi-turn conversations produced by generate_multiturn_dataset.py.
#
# Full pipeline order:
#   generate_dataset.py          (Stage 0 — PDF → SFT data)
#   SLM_finetune_v2.py           (Stage 1 — single-turn SFT)
#   generate_multiturn_dataset.py (Stage 2a — expand to multi-turn)
#   multiturn_train.py           (Stage 2b — multi-turn SFT)  ← HERE
#   generate_dpo_pairs.py        (Stage 3a — create DPO pairs)
#   mistral_dpo_train.py         (Stage 3b — DPO training)
# =============================================================


# =============================================================
# 0) CONFIG  ← the only section your team needs to edit
# =============================================================
import os as _os

# Your single-turn SFT training dataset (Alpaca-format JSONL —
# same file used by the SFT training script).
DATASET_PATH        = _os.environ.get("SFT_DATASET_PATH", "training_dataset.jsonl")
MULTITURN_OUT_PATH  = "multiturn_dataset.jsonl"
COMBINED_OUT_PATH   = "combined_sft_multiturn.jsonl"   # SFT + multi-turn merged

# The SFT (V1) adapter this stage trains on top of.
# Set to YOUR org's HF repo produced by the SFT script,
# e.g. "your-org/mistral-sft-v1". A local adapter directory
# from the same session is preferred automatically (see below).
SFT_MODEL_PATH      = _os.environ.get("SFT_MODEL_PATH", "your-org/mistral-sft-v1")
MULTITURN_SAVE_PATH = "mistral_multiturn_lora"
OUTPUT_DIR          = "mistral_multiturn_outputs"

# ── HuggingFace Hub ──────────────────────────────────────────
# Trained adapters are pushed to the account that owns HF_TOKEN.
# Use a WRITE token from YOUR account/org (Settings → Access Tokens).
# Export before running — never hardcode tokens in source:
#   export HF_TOKEN=hf_xxxx
#   export HUB_MODEL_ID=your-org/mistral-multiturn
HUB_TOKEN           = _os.environ.get("HF_TOKEN", "")
HUB_MODEL_ID        = _os.environ.get("HUB_MODEL_ID", "")
PUSH_TO_HUB         = bool(HUB_TOKEN and HUB_MODEL_ID)

MAX_SEQ_LENGTH      = 4096
LOAD_IN_4BIT        = True

# Training
LR              = 2e-4
EPOCHS          = 2
PER_DEVICE_BS   = 2
GRAD_ACC_STEPS  = 4
WARMUP_RATIO    = 0.1
SEED            = 3407

# ⚠️ DOMAIN-SPECIFIC — REPLACE FOR YOUR USE CASE and keep it IDENTICAL
# to the prompt used in the SFT script and at inference time.
SYSTEM_PROMPT = (
    "You are an expert technical support specialist with 10+ years of experience "
    "in your domain. You provide detailed, step-by-step diagnostic and repair "
    "guidance. Your responses always: (1) identify the likely root cause, "
    "(2) provide numbered diagnostic steps from least invasive to most invasive, "
    "(3) note when professional service is required, and (4) cite the source of your "
    "information. You never guess — you reason systematically."
)


# =============================================================
# 1) IMPORTS
# =============================================================
import os
import json
import math
from pathlib import Path

# Stub for missing transformers symbol (sentence-transformers can downgrade transformers)
import transformers.utils as _tu
if not hasattr(_tu, "is_bitsandbytes_multi_backend_available"):
    _tu.is_bitsandbytes_multi_backend_available = lambda: False

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Transformers 4.56.x bug fix: list_repo_templates() raises RemoteEntryNotFoundError
# for repos that don't have an additional_chat_templates folder (all unsloth quant models).
try:
    import transformers.utils.hub as _hub_mod
    import transformers.tokenization_utils_base as _tok_mod
    _orig_lrt = _hub_mod.list_repo_templates
    def _safe_lrt(repo_id, *a, **kw):
        try: return _orig_lrt(repo_id, *a, **kw)
        except Exception: return []
    _hub_mod.list_repo_templates = _safe_lrt
    if hasattr(_tok_mod, "list_repo_templates"):
        _tok_mod.list_repo_templates = _safe_lrt
    print("✅ Applied transformers 4.56.x list_repo_templates patch")
except Exception:
    pass


# =============================================================
# 2) MODEL LOADING + TRAINING FORMAT
# =============================================================
def load_model():
    """Load the single-turn SFT adapter to continue training on.

    Prefers a local adapter directory from the same session
    (mistral_v1_lora); falls back to SFT_MODEL_PATH on the HF Hub.
    """
    print("\n✅ Loading model for multi-turn SFT...")
    _v1_local = "mistral_v1_lora"
    _load_path = _v1_local if os.path.exists(_v1_local) else SFT_MODEL_PATH
    print(f"✅ Loading V1 adapter from: {_load_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=_load_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_multiturn(batch):
    """
    Convert ShareGPT-style conversations to Mistral instruct format.
    <s>[INST] system\n\nuser1 [/INST] assistant1</s>[INST] user2 [/INST] assistant2</s>
    """
    texts = []
    for conv_list in batch["conversations"]:
        system_text = ""
        user_turns  = []   # list of (user_msg, assistant_msg) pairs
        pending_user = None

        for turn in conv_list:
            role  = turn.get("from", "")
            value = (turn.get("value", "") or "").strip()
            if role == "system":
                system_text = value
            elif role in ["human", "user"]:
                pending_user = value
            elif role in ["gpt", "assistant"]:
                user_turns.append((pending_user or "", value))
                pending_user = None

        chat = ""
        for i, (usr, ast) in enumerate(user_turns):
            if i == 0 and system_text:
                chat += f"<s>[INST] {system_text}\n\n{usr} [/INST] {ast}</s>"
            else:
                chat += f"[INST] {usr} [/INST] {ast}</s>"

        texts.append(chat.strip())
    return {"text": texts}


def prepare_datasets():
    """Load the generated multi-turn JSONL, split 90/10, format for SFT."""
    full_ds = load_dataset("json", data_files=MULTITURN_OUT_PATH, split="train")
    full_ds = full_ds.shuffle(seed=SEED)

    split_ds = full_ds.train_test_split(test_size=0.10, seed=SEED)
    train_ds = split_ds["train"]
    eval_ds  = split_ds["test"]

    print(f"\n✅ Dataset split: {len(train_ds)} train / {len(eval_ds)} eval")

    keep_cols = ["conversations"]
    train_fmt = train_ds.select_columns(keep_cols).map(
        format_multiturn, batched=True, remove_columns=keep_cols)
    eval_fmt  = eval_ds.select_columns(keep_cols).map(
        format_multiturn, batched=True, remove_columns=keep_cols)

    print(f"\n✅ Sample formatted conversation (first 800 chars):")
    print(train_fmt["text"][0][:800])
    print("...")
    return train_fmt, eval_fmt, len(train_ds)


# =============================================================
# 3) TRAIN + SAVE
# =============================================================
def train_and_save(model, tokenizer, train_fmt, eval_fmt, n_train):
    """Run multi-turn SFT, save the best adapter locally, push to Hub."""
    steps_per_epoch = math.ceil(n_train / (PER_DEVICE_BS * GRAD_ACC_STEPS))
    total_steps     = steps_per_epoch * EPOCHS
    eval_steps      = max(steps_per_epoch // 2, 10)

    print(f"\n✅ Starting multi-turn SFT training...")
    print(f"   lr         = {LR:.1e}")
    print(f"   epochs     = {EPOCHS}")
    print(f"   steps      = {total_steps}")
    print(f"   eval every = {eval_steps} steps")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_fmt,
        eval_dataset=eval_fmt,
        dataset_text_field="text",
        packing=True,
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BS,
            per_device_eval_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            optim="adamw_8bit",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=SEED,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\n📊 FINAL MULTI-TURN METRICS")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    if "eval_loss" in metrics:
        print(f"\n🔥 Perplexity: {math.exp(metrics['eval_loss']):.4f}")

    print(f"\n✅ Saving multi-turn LoRA → {MULTITURN_SAVE_PATH}")
    model.save_pretrained(MULTITURN_SAVE_PATH)
    tokenizer.save_pretrained(MULTITURN_SAVE_PATH)

    if PUSH_TO_HUB and HUB_MODEL_ID and not HUB_MODEL_ID.startswith("your-"):
        from huggingface_hub import login
        login(token=HUB_TOKEN)
        print(f"\n✅ Pushing multi-turn adapters → {HUB_MODEL_ID}")
        model.push_to_hub(HUB_MODEL_ID)
        tokenizer.push_to_hub(HUB_MODEL_ID)
        print(f"✅ Done — https://huggingface.co/{HUB_MODEL_ID}")

    return metrics


# =============================================================
# 4) SMOKE TEST — multi-turn inference
# =============================================================
def build_multiturn_prompt(history):
    """Build a Mistral-instruct prompt from (role, text) history pairs."""
    prompt = ""
    for i, (role, text) in enumerate(history):
        if role == "user":
            if i == 0:
                prompt += f"<s>[INST] {SYSTEM_PROMPT}\n\n{text} [/INST]"
            else:
                prompt += f"[INST] {text} [/INST]"
        else:
            prompt += f" {text}</s>"
    return prompt


def smoke_test(model, tokenizer):
    """Quick 2-turn conversation to sanity-check follow-up handling."""
    print("\n\n" + "=" * 62)
    print("  SMOKE TEST — Multi-turn conversation")
    print("=" * 62)

    FastLanguageModel.for_inference(model)

    # Replace with a realistic multi-turn scenario from YOUR domain
    conversation_history = [
        ("user", _os.environ.get(
            "SMOKE_TEST_QUESTION",
            "The unit powers on but shuts down intermittently after a few minutes.")),
    ]

    print("\n── Turn 1 ──────────────────────────────────────────────")
    print(f"User: {conversation_history[0][1]}")

    for turn_num in range(2):
        prompt = build_multiturn_prompt(conversation_history)
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=1500).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"\nAssistant: {response[:300]}...")
        conversation_history.append(("assistant", response))

        # Simulate user follow-up
        if turn_num == 0:
            followup = "I already tried the first step you suggested. The problem is still there."
            conversation_history.append(("user", followup))
            print(f"\n── Turn {turn_num + 2} ──────────────────────────────────────────────")
            print(f"User: {followup}")


# =============================================================
# 5) MAIN
# =============================================================
def main():
    # Fail early with a clear message if the multi-turn dataset hasn't been generated yet
    if not Path(MULTITURN_OUT_PATH).exists():
        raise FileNotFoundError(
            f"\n❌ Multi-turn dataset not found: {MULTITURN_OUT_PATH}\n\n"
            f"   Run Stage 2a first:\n"
            f"     python generate_multiturn_dataset.py\n\n"
            f"   Or set MULTITURN_OUT_PATH to an existing file."
        )

    model, tokenizer = load_model()
    train_fmt, eval_fmt, n_train = prepare_datasets()
    train_and_save(model, tokenizer, train_fmt, eval_fmt, n_train)
    smoke_test(model, tokenizer)

    print("\n\n🔥 MULTI-TURN TRAINING COMPLETE")
    print(f"   Model saved : {MULTITURN_SAVE_PATH}")
    print(f"\n   Next: python generate_dpo_pairs.py")


if __name__ == "__main__":
    main()
