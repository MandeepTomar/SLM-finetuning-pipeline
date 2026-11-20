import torch
assert torch.cuda.is_available(), "Enable GPU runtime"

# =============================================================
# MISTRAL DPO TRAINING
# Trains on top of your multi-turn SFT adapter using DPO.
#
# ⚠️ PREREQUISITE — STAGE 3 OF 3 (final stage). Both SFT stages
# must be complete first; SFT_MODEL_PATH must point to the
# multi-turn adapter. DPO only re-weights behaviour the SFT
# stages already taught — it cannot teach new knowledge, so
# running it on an untrained model produces nothing useful.
# Order: SFT → multi-turn → DPO (this script).
#
# DPO loss:
#   log σ(β * log[π(chosen|x)/π_ref(chosen|x)]
#              - β * log[π(rejected|x)/π_ref(rejected|x)])
#
# The reference model (π_ref) is a frozen copy of the SFT model.
# DPO shifts the model toward chosen (software-first, structured)
# responses and away from rejected (hardware-first, incomplete).
#
# Key difference from SFT: we're not teaching WHAT to say,
# we're teaching WHICH style to prefer.
#
# COMPLEXITY:
#   Time  : ~4× SFT per pair — each step runs 4 forward passes
#           (policy + frozen reference, on chosen + rejected),
#           each O(L²) attention with L = MAX_LENGTH. Reducing
#           MAX_LENGTH 2048→1024 gives ~4× faster attention.
#   Space : policy (4-bit, ~4 GB) + frozen reference copy +
#           activations for both sequences — this is why
#           PER_DEVICE_BS=1 with GRAD_ACC_STEPS=8 on a 16 GB T4.
# =============================================================


# =============================================================
# 0) CONFIG  ← the only section your team needs to edit
# =============================================================
import os as _os

# The multi-turn SFT adapter this stage trains on top of —
# YOUR org's HF repo produced by multiturn_train.py.
SFT_MODEL_PATH  = _os.environ.get("SFT_MODEL_PATH", "your-org/mistral-multiturn")

# DPO preference pairs: JSONL with instruction/input/chosen/rejected
# fields. "chosen" = the response style you want (e.g. structured,
# least-invasive-first); "rejected" = the style to move away from.
DPO_PAIRS_PATH  = _os.environ.get("DPO_PAIRS_PATH", "dpo_pairs.jsonl")
DPO_SAVE_PATH   = "mistral_dpo_lora"
OUTPUT_DIR      = "mistral_dpo_outputs"

# ── HuggingFace Hub ──────────────────────────────────────────
# Adapters are pushed to the account that owns HF_TOKEN.
# Export before running — never hardcode tokens in source:
#   export HF_TOKEN=hf_xxxx
#   export HUB_MODEL_ID=your-org/mistral-dpo
HUB_TOKEN       = _os.environ.get("HF_TOKEN", "")
HUB_MODEL_ID    = _os.environ.get("HUB_MODEL_ID", "")
PUSH_TO_HUB     = bool(HUB_TOKEN and HUB_MODEL_ID)

MAX_SEQ_LENGTH  = 4096
LOAD_IN_4BIT    = True

BETA            = 0.5       # raised from 0.2 — previous run collapsed (loss=0, rewards/chosen=162)
                            # higher β tightens the KL constraint and prevents reward divergence
LOSS_TYPE       = "ipo"     # IPO loss is more robust than sigmoid DPO when pairs are similar
                            # IPO does not suffer from the reward-hacking collapse seen with sigmoid

LR              = 2e-5      # halved from 5e-5 — slower updates reduce overfitting to easy pairs
EPOCHS          = 1
PER_DEVICE_BS   = 1
GRAD_ACC_STEPS  = 8
WARMUP_RATIO    = 0.1
MAX_LENGTH      = 1024      # reduced from 2048 — forces model to focus on response structure
MAX_PROMPT_LEN  = 384       # reduced from 512

LORA_R          = 4         # reduced from 8 — less capacity means less overfitting to easy signal
LORA_ALPHA      = 8
LORA_DROPOUT    = 0.05      # small dropout as additional regularisation

SEED            = 3407

# ⚠️ DOMAIN-SPECIFIC — REPLACE FOR YOUR USE CASE and keep it IDENTICAL
# to the prompt used at inference time in your system. The structural
# rules (software-first, ordered escalation) are what DPO reinforces.
SYSTEM_PROMPT = (
    "You are an expert repair technician with 10+ years of experience. "
    "STRICT RULE: Always exhaust ALL software and settings fixes before suggesting any hardware repair or disassembly. "
    "Your response structure must ALWAYS follow this order: "
    "(1) Identify the likely root cause. "
    "(2) Software fixes first — settings, restarts, updates, resets. "
    "(3) External hardware checks — cable, charger, accessories. "
    "(4) Physical inspection — port cleaning, visible damage. "
    "(5) Disassembly and internal repair ONLY as the last resort. "
    "(6) State clearly when professional service is required. "
    "Never open a device until all software options are exhausted. "
    "If the user says they completed a step, acknowledge it and give the NEXT step in sequence."
)


# =============================================================
# 1) IMPORTS + PATCHES
# =============================================================
import json
import math
import numpy as np
from datasets import load_dataset, Dataset

# ── Patch: transformers 4.56.x list_repo_templates 404 bug ───
try:
    import transformers.utils.hub as _hub
    _orig = getattr(_hub, "list_repo_templates", None)
    if _orig:
        def _safe(*a, **kw):
            try: return _orig(*a, **kw)
            except Exception: return []
        _hub.list_repo_templates = _safe

    import transformers.tokenization_utils_base as _tub
    _orig2 = getattr(_tub, "list_repo_templates", None)
    if _orig2:
        def _safe2(*a, **kw):
            try: return _orig2(*a, **kw)
            except Exception: return []
        _tub.list_repo_templates = _safe2
except Exception:
    pass

from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig


# =============================================================
# 2) LOAD SFT MODEL
# =============================================================
def load_model():
    """Load the multi-turn SFT adapter that DPO will refine."""
    print("\n" + "=" * 62)
    print("  MISTRAL DPO TRAINING — on top of the multi-turn SFT adapter")
    print("=" * 62)
    print(f"\n✅ Loading SFT base model: {SFT_MODEL_PATH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model.print_trainable_parameters()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model ready for DPO")
    return model, tokenizer


# =============================================================
# 3) LOAD + CONVERT DPO PAIRS
# The pairs file has prompts in Alpaca format
# (### System / ### Instruction / ### Response).
# We rebuild the prompt field in Mistral [INST] format here.
# The chosen/rejected response text is format-agnostic — reuse as-is.
# =============================================================
def convert_to_mistral_format(batch):
    """Replace the Alpaca-format prompt with a Mistral [INST] prompt."""
    new_prompts = []
    for instruction, inp in zip(batch["instruction"], batch["input"]):
        instruction = (instruction or "").strip()
        inp         = (inp or "").strip()
        user_text   = f"{instruction}\n\n{inp}".strip() if inp else instruction
        # Mistral prompt ends at [/INST] — DPOTrainer appends chosen/rejected after
        new_prompts.append(f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_text} [/INST]")
    return {"prompt": new_prompts}


def prepare_datasets():
    """Load preference pairs, convert to Mistral format, split 90/10."""
    print(f"\n✅ Loading DPO pairs: {DPO_PAIRS_PATH}")

    raw_ds = load_dataset("json", data_files=DPO_PAIRS_PATH, split="train")
    print(f"   Total pairs loaded : {len(raw_ds)}")

    converted_ds = raw_ds.map(
        convert_to_mistral_format,
        batched=True,
        desc="Converting prompts to Mistral format",
    )

    # Keep only the columns DPOTrainer needs
    converted_ds = converted_ds.select_columns(["prompt", "chosen", "rejected"])
    print(f"   After conversion   : {len(converted_ds)} pairs")

    split_ds  = converted_ds.train_test_split(test_size=0.10, seed=SEED)
    train_ds  = split_ds["train"].shuffle(seed=SEED)
    eval_ds   = split_ds["test"].shuffle(seed=SEED)

    print(f"   Train pairs : {len(train_ds)}")
    print(f"   Eval pairs  : {len(eval_ds)}")

    eval_ds.to_json("dpo_eval_set.jsonl")
    print("💾 Saved DPO eval set → dpo_eval_set.jsonl")

    sample = train_ds[0]
    print(f"\n── Sample DPO pair (Mistral format) ────────────────────────")
    print(f"Prompt (last 200 chars): ...{sample['prompt'][-200:]}")
    print(f"\nChosen (first 200 chars):\n{sample['chosen'][:200]}...")
    print(f"\nRejected (first 200 chars):\n{sample['rejected'][:200]}...")
    return train_ds, eval_ds


# =============================================================
# 4) TRAIN + EVALUATE
# =============================================================
def train(model, tokenizer, train_ds, eval_ds):
    """Run DPO training and print preference-learning metrics."""
    steps_per_epoch = math.ceil(len(train_ds) / (PER_DEVICE_BS * GRAD_ACC_STEPS))
    total_steps     = steps_per_epoch * EPOCHS
    eval_steps      = max(steps_per_epoch // 2, 10)

    print(f"\n✅ Training schedule:")
    print(f"   Steps per epoch : {steps_per_epoch}")
    print(f"   Total steps     : {total_steps}")
    print(f"   Eval every      : {eval_steps} steps")
    print(f"   β (beta)        = {BETA}")
    print(f"   lr              = {LR:.1e}")

    print(f"\n✅ Starting DPO training...")

    trainer = DPOTrainer(
    model=model,
    ref_model=None,       # TRL auto-creates frozen reference copy
    tokenizer=tokenizer,

    train_dataset=train_ds,
    eval_dataset=eval_ds,

    args=DPOConfig(
        beta=BETA,
        loss_type=LOSS_TYPE,

        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LEN,

        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,

        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",

        logging_steps=5,

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
            remove_unused_columns=True,
        ),
    )

    trainer.train()

    # ── DPO metrics ──────────────────────────────────────────
    print("\n✅ Running final evaluation...")
    metrics = trainer.evaluate()

    print("\n📊 FINAL DPO METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\n📌 DPO Metric Interpretation:")
    if "eval_rewards/margins" in metrics:
        margin = metrics["eval_rewards/margins"]
        print(f"  Reward margin : {margin:.4f}  ", end="")
        if margin > 0.5:
            print("✅ Strong preference separation")
        elif margin > 0.1:
            print("✅ Moderate preference learning")
        else:
            print("⚠️  Weak separation — consider lower β or more epochs")

    if "eval_rewards/chosen" in metrics and "eval_rewards/rejected" in metrics:
        ch = metrics["eval_rewards/chosen"]
        rj = metrics["eval_rewards/rejected"]
        print(f"  Chosen reward  : {ch:.4f}  (target: positive ↑)")
        print(f"  Rejected reward: {rj:.4f}  (target: negative ↓)")

    return metrics


# =============================================================
# 5) SAVE + PUSH
# =============================================================
def save_and_push(model, tokenizer):
    """Save the DPO adapter locally and push to the Hub if configured."""
    print(f"\n✅ Saving DPO LoRA adapters → {DPO_SAVE_PATH}")
    model.save_pretrained(DPO_SAVE_PATH)
    tokenizer.save_pretrained(DPO_SAVE_PATH)

    if PUSH_TO_HUB:
        from huggingface_hub import login
        login(token=HUB_TOKEN)
        print(f"\n✅ Pushing → {HUB_MODEL_ID}")
        model.push_to_hub(HUB_MODEL_ID)
        tokenizer.push_to_hub(HUB_MODEL_ID)
        print(f"✅ https://huggingface.co/{HUB_MODEL_ID}")


# =============================================================
# 6) SMOKE TEST — verify the DPO objective took hold
# =============================================================
def smoke_test(model, tokenizer):
    """Check that responses start non-invasive and don't jump to disassembly."""
    print("\n\n" + "=" * 62)
    print("  SMOKE TEST — SFT vs DPO response style comparison")
    print("=" * 62)

    FastLanguageModel.for_inference(model)

    # Replace with test questions from YOUR domain. Each pair is
    # (question, expected behaviour). The keyword checks below verify the
    # DPO objective: responses should start with non-invasive/software
    # steps and not jump straight to disassembly.
    test_cases = [
        ("The unit powers on but shuts down randomly after a few minutes.",
         "starts with software/settings checks"),
        ("The display flickers and sometimes goes black during operation.",
         "software check before hardware"),
        ("The device has become extremely slow after two years of use.",
         "clears caches / software maintenance first"),
    ]

    for question, expected_behaviour in test_cases:
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nDiagnose and fix the issue\n\n{question} [/INST]"

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=MAX_PROMPT_LEN).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Adjust these keyword lists to your domain's vocabulary
        starts_software = any(kw in decoded[:300].lower() for kw in [
            "possible cause", "step 1: check", "settings", "software",
            "restart", "update", "cache", "reset",
        ])
        jumps_hardware = any(kw in decoded[:300].lower() for kw in [
            "screwdriver", "heat gun", "pry open",
            "disassemble", "opening tool",
        ])

        print(f"\n── {question[:70]}...")
        print(f"   Expected : {expected_behaviour}")
        print(f"   Response (first 300): {decoded[:300]}...")
        print(f"   ✅ Software-first : {starts_software}")
        print(f"   ⚠️  Hardware jump  : {jumps_hardware}")


# =============================================================
# 7) MAIN
# =============================================================
def main():
    model, tokenizer = load_model()
    train_ds, eval_ds = prepare_datasets()
    train(model, tokenizer, train_ds, eval_ds)
    save_and_push(model, tokenizer)
    smoke_test(model, tokenizer)

    print("\n\n🔥 DPO TRAINING COMPLETE")
    print(f"   SFT model : {SFT_MODEL_PATH}")
    print(f"   DPO model : {DPO_SAVE_PATH}")
    print(f"\n   Next: evaluate both adapters on your held-out eval set and compare")
    print(f"   domain metrics (root cause accuracy, step ordering, hallucination, safety)")


if __name__ == "__main__":
    main()
