"""
generate_dpo_pairs.py — Stage 3a of the SLM fine-tuning pipeline.

Generates DPO preference pairs from your trained SFT model using
self-play rejection: the model is run once WITH the system prompt
(chosen = ideal, structured response) and once WITHOUT it (rejected =
degraded response that misses the ordering rules).

This approach is domain-agnostic: "wrong" is defined entirely by the
system prompt's ordering rules, not by hardcoded domain knowledge.
The model naturally reverts to unstructured, pattern-matched responses
when the system prompt is removed — exactly the behaviour DPO should penalise.

Pipeline position:
    multiturn_train.py           (Stage 2b — multi-turn SFT)
          │
          ▼  mistral_multiturn_lora / your-org/mistral-multiturn
    generate_dpo_pairs.py          ← YOU ARE HERE  (Stage 3a)
          │
          ▼  dpo_pairs.jsonl
    mistral_dpo_train.py         (Stage 3b — DPO training)

GPU required — loads the trained SFT model for inference.

Required environment variables:
    SFT_MODEL_PATH    — HF repo or local path of the Stage 1 or 2 adapter
                        (default: mistral_multiturn_lora)
    SFT_DATASET_PATH  — Alpaca-format JSONL to generate pairs from
                        (default: training_dataset.jsonl)
    DPO_PAIRS_PATH    — output path (default: dpo_pairs.jsonl)

Optional:
    MAX_NEW_TOKENS    — max tokens per rejected response (default: 400)
    BATCH_SIZE        — inference batch size (default: 4, reduce if OOM)

⚠️ DOMAIN-SPECIFIC:
    SYSTEM_PROMPT — must be IDENTICAL to the one in all other pipeline scripts.
    Copy it from generate_dataset.py without changes.
"""

import torch
assert torch.cuda.is_available(), "GPU required — enable CUDA runtime"

import os
import json
import random
import statistics
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel

random.seed(42)

# =============================================================
# 0) CONFIG  ← set via environment variables, not here
# =============================================================
SFT_MODEL_PATH  = os.environ.get("SFT_MODEL_PATH",   "mistral_multiturn_lora")
DATASET_PATH    = os.environ.get("SFT_DATASET_PATH",  "training_dataset.jsonl")
DPO_PAIRS_PATH  = os.environ.get("DPO_PAIRS_PATH",    "dpo_pairs.jsonl")

MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "400"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",     "4"))

MIN_REJECTED_LEN = 100   # chars — reject if model collapses to a fragment

# ⚠️ DOMAIN-SPECIFIC — must be IDENTICAL to all other pipeline scripts
# and to your inference system. Copy from generate_dataset.py exactly.
SYSTEM_PROMPT = (
    "You are an expert technical support specialist with 10+ years of experience "
    "in your domain. You provide detailed, step-by-step diagnostic and repair "
    "guidance. Your responses always: (1) identify the likely root cause, "
    "(2) provide numbered diagnostic steps from least invasive to most invasive, "
    "(3) note when professional service is required, and (4) cite the source of your "
    "information. You never guess — you reason systematically."
)

# =============================================================
# 1) PROMPT BUILDERS
# =============================================================
def build_prompt_with_system(row: dict) -> str:
    """Full Mistral instruct prompt — used for the CHOSEN response."""
    instruction = (row.get("instruction") or "").strip()
    inp         = (row.get("input")       or "").strip()
    user_text   = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_text} [/INST]"


def build_prompt_without_system(row: dict) -> str:
    """
    Prompt WITHOUT system prompt — produces the REJECTED response.

    Why this works: the system prompt is what enforces structured, least-
    invasive-first reasoning. Without it the model reverts to its pre-training
    prior — typically pattern-matched, unstructured responses that jump to
    conclusions without systematic diagnosis. That is exactly the behaviour
    DPO should move the model away from.

    This is more principled than hand-writing "bad" responses: the rejected
    outputs are genuine model-generated degradations, not artificial ones.
    """
    instruction = (row.get("instruction") or "").strip()
    inp         = (row.get("input")       or "").strip()
    user_text   = f"{instruction}\n\n{inp}".strip() if inp else instruction
    # Plain Alpaca format — no system persona
    return (
        f"Below is an instruction that describes a task, paired with an input "
        f"that provides further context.\n\n"
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )

# =============================================================
# 2) QUALITY FILTER
# =============================================================
def is_valid_pair(chosen: str, rejected: str) -> tuple:
    """Return (True, "") or (False, reason)."""
    if len(rejected) < MIN_REJECTED_LEN:
        return False, "rejected_too_short"
    if rejected[:200].strip() == chosen[:200].strip():
        return False, "rejected_identical_to_chosen"
    # Both should be non-trivial
    if len(chosen.split()) < 30:
        return False, "chosen_too_short"
    return True, ""

# =============================================================
# 3) MAIN
# =============================================================
def main():
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"SFT dataset not found: {DATASET_PATH}\n"
            f"Set SFT_DATASET_PATH or run generate_dataset.py first."
        )

    print("\n" + "=" * 62)
    print("  STAGE 3a — DPO PREFERENCE PAIR GENERATOR")
    print("  Strategy: self-play rejection (remove system prompt)")
    print("=" * 62)

    # ── Load model ────────────────────────────────────────────
    # Prefer local adapter from same session; fall back to SFT_MODEL_PATH.
    local_candidates = ["mistral_multiturn_lora", "mistral_v1_lora"]
    load_path = SFT_MODEL_PATH
    for candidate in local_candidates:
        if os.path.exists(candidate):
            load_path = candidate
            break

    print(f"\n✅ Loading SFT model: {load_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model ready for inference")

    # ── Load dataset ──────────────────────────────────────────
    print(f"\n✅ Loading dataset: {DATASET_PATH}")
    data = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"   Total entries: {len(data)}")
    random.shuffle(data)

    # ── Generate rejected responses (batched) ─────────────────
    print(f"\n✅ Generating rejected responses (WITHOUT system prompt)...")
    print(f"   Batch size  : {BATCH_SIZE}")
    print(f"   Max tokens  : {MAX_NEW_TOKENS}")
    print(f"   Samples     : {len(data)}")

    prompts_rejected = [build_prompt_without_system(row) for row in data]
    rejected_responses = []

    for i in range(0, len(prompts_rejected), BATCH_SIZE):
        batch = prompts_rejected[i: i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH // 2,   # leave room for generation
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,      # higher than training — want varied/degraded outputs
                top_p=0.9,
                repetition_penalty=1.0,   # no penalty — let it repeat; that's realistic degradation
                pad_token_id=tokenizer.eos_token_id,
            )

        for out_ids, inp_ids in zip(output_ids, inputs["input_ids"]):
            new_ids  = out_ids[len(inp_ids):]
            rejected = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            rejected_responses.append(rejected)

        done = min(i + BATCH_SIZE, len(data))
        print(f"   Generated: {done}/{len(data)}", end="\r")

    print(f"\n✅ {len(rejected_responses)} rejected responses generated")

    # ── Filter + build pairs ──────────────────────────────────
    print("\n✅ Filtering pairs...")
    dpo_entries        = []
    skip_counts: dict  = {}

    for row, rejected in zip(data, rejected_responses):
        chosen = row.get("output", "")
        valid, reason = is_valid_pair(chosen, rejected)
        if not valid:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue

        dpo_entries.append({
            # Core DPO fields (what mistral_dpo_train.py reads)
            "instruction" : row.get("instruction", "Diagnose and fix the issue"),
            "input"       : row.get("input",        ""),
            "chosen"      : chosen,
            "rejected"    : rejected,
            # Metadata — not used in training, useful for analysis
            "topic"       : row.get("topic",        ""),
            "source"      : row.get("source",       ""),
            "difficulty"  : row.get("difficulty",   ""),
        })

    print(f"   Total pairs     : {len(data)}")
    for reason, n in sorted(skip_counts.items(), key=lambda x: -x[1]):
        print(f"   Skipped ({reason}) : {n}")
    print(f"   Kept            : {len(dpo_entries)}")

    if not dpo_entries:
        print("\n⚠️  No pairs survived filtering. Possible causes:")
        print("   - Model is too good without a system prompt (all outputs identical to chosen)")
        print("   - BATCH_SIZE too large causing tokenizer truncation")
        print("   - MAX_NEW_TOKENS too low — rejected responses are empty")
        return

    # ── Save ──────────────────────────────────────────────────
    with open(DPO_PAIRS_PATH, "w", encoding="utf-8") as f:
        for entry in dpo_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n💾 Saved {len(dpo_entries)} DPO pairs → {DPO_PAIRS_PATH}")

    # ── Sample inspection ─────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SAMPLE PAIRS — inspect before running DPO training")
    print("=" * 62)

    for i, entry in enumerate(dpo_entries[:3]):
        print(f"\n── Pair #{i+1} (topic: {entry.get('topic', '?')}) ──")
        print(f"Input    : {entry['input'][:120]}")
        print(f"\nCHOSEN (first 300 chars):\n  {entry['chosen'][:300]}...")
        print(f"\nREJECTED (first 300 chars):\n  {entry['rejected'][:300]}...")

        # Check for the key quality signal: does chosen start structured,
        # rejected start unstructured?
        chosen_structured  = any(kw in entry["chosen"][:200].lower()  for kw in [
            "issue:", "possible cause", "step 1:", "root cause", "solution:"
        ])
        rejected_unstructured = not any(kw in entry["rejected"][:200].lower() for kw in [
            "issue:", "possible cause", "step 1:", "root cause", "solution:"
        ])
        print(f"\n  ✅ Chosen is structured  : {chosen_structured}")
        print(f"  ✅ Rejected is degraded  : {rejected_unstructured}")

    # ── Stats ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  DATASET STATS")
    print("=" * 62)

    topic_counts: dict = {}
    for e in dpo_entries:
        t = e.get("topic", "other")
        topic_counts[t] = topic_counts.get(t, 0) + 1

    chosen_lens   = [len(e["chosen"])   for e in dpo_entries]
    rejected_lens = [len(e["rejected"]) for e in dpo_entries]

    print(f"\n  Total pairs           : {len(dpo_entries)}")
    print(f"  Chosen avg length     : {statistics.mean(chosen_lens):.0f} chars")
    print(f"  Rejected avg length   : {statistics.mean(rejected_lens):.0f} chars")
    print(f"  (Rejected should be shorter and less structured than chosen)")

    if topic_counts:
        print(f"\n  Top topics:")
        for t, n in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {t:<30} {n}")

    print(f"\n✅ Ready for DPO training")
    print(f"   Next: export DPO_PAIRS_PATH={DPO_PAIRS_PATH}")
    print(f"          python mistral_dpo_train.py")


if __name__ == "__main__":
    main()
