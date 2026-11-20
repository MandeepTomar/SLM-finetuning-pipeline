"""
generate_multiturn_dataset.py — Stage 2a of the SLM fine-tuning pipeline.

Reads the single-turn SFT dataset (training_dataset.jsonl from Stage 0),
expands each Q→A pair into up to 6 realistic multi-turn conversation
variants, and writes multiturn_dataset.jsonl — the file that
multiturn_train.py (Stage 2b) expects as input.

Pipeline position:
    generate_dataset.py          (Stage 0 — PDF → SFT data)
          │
          ▼  training_dataset.jsonl
    SLM_finetune_v2.py           (Stage 1 — single-turn SFT)
          │
          ▼  mistral_sft_lora / your-org/mistral-sft-v1
    generate_multiturn_dataset.py  ← YOU ARE HERE  (Stage 2a)
          │
          ▼  multiturn_dataset.jsonl
    multiturn_train.py           (Stage 2b — multi-turn SFT)
          │
          ▼
    generate_dpo_pairs.py        (Stage 3a — DPO pair generation)
          │
          ▼  dpo_pairs.jsonl
    mistral_dpo_train.py         (Stage 3b — DPO training)

No GPU required — this is pure Python string processing.
Runs in seconds even for thousands of entries.

Required environment variables:
    SFT_DATASET_PATH   — path to the Alpaca-format JSONL from Stage 0
                         (default: training_dataset.jsonl)
    MULTITURN_OUT_PATH — output path (default: multiturn_dataset.jsonl)

⚠️ DOMAIN-SPECIFIC — the following sections must be rewritten for your domain:
    1. make_clarification_conversation — SPECIFIC_IDENTIFIERS and clarification
       question/answer pairs (what to ask for when model isn't specified)
    2. make_success_confirmation_conversation — PREVENTION_MAP
       (topic → prevention advice; keys must match your dataset's "topic" field)
    3. make_followup_question_conversation — FOLLOWUP_MAP
       (topic → (follow-up question, expert answer) pairs)
    4. SYSTEM_PROMPT — must be IDENTICAL to the one in all training scripts
       and at inference time
"""

import os
import json
import random
from pathlib import Path

random.seed(42)

# =============================================================
# 0) CONFIG
# =============================================================
DATASET_PATH       = os.environ.get("SFT_DATASET_PATH", "training_dataset.jsonl")
MULTITURN_OUT_PATH = os.environ.get("MULTITURN_OUT_PATH", "multiturn_dataset.jsonl")

# ⚠️ DOMAIN-SPECIFIC — must be IDENTICAL across all pipeline scripts
# and at inference time. Copy from generate_dataset.py / SLM_finetune_v2.py.
SYSTEM_PROMPT = (
    "You are an expert technical support specialist with 10+ years of experience "
    "in your domain. You provide detailed, step-by-step diagnostic and repair "
    "guidance. Your responses always: (1) identify the likely root cause, "
    "(2) provide numbered diagnostic steps from least invasive to most invasive, "
    "(3) note when professional service is required, and (4) cite the source of your "
    "information. You never guess — you reason systematically."
)

# =============================================================
# ⚠️ DOMAIN-SPECIFIC MAPS — rewrite for your domain
# =============================================================

# PREVENTION_MAP: topic (from your dataset's "topic" field) → advice to give
# when the user says "that worked, how do I prevent this in future?"
# Keys must match the "topic" values your generate_dataset.py produces.
# The fallback (None key) is used for any unrecognised topic.
PREVENTION_MAP = {
    # ── Example entries — replace with your domain's topics ──
    # "battery":   "Avoid charging above 40 °C, keep the device between 20-80% where possible, ...",
    # "display":   "Fit a screen protector, and re-seat the digitizer flex whenever the panel is opened...",
    # "charging":  "Inspect and clear the charging port of lint at every service...",
    None: (
        "Keep your system software and firmware updated, follow the manufacturer's "
        "recommended maintenance schedule, and document any recurring error codes "
        "for your next service appointment."
    ),
}

# FOLLOWUP_MAP: topic → (follow-up question the user might ask, expert answer)
# Only entries whose topic key appears here will produce followup_question turns.
# Leave empty ({}) if you haven't written domain follow-ups yet — those turns
# will simply be skipped (no error).
FOLLOWUP_MAP = {
    # ── Example entries — replace with your domain's topics ──
    # "battery": (
    #     "What does the battery health report actually tell me?",
    #     "It reports maximum capacity against the original design capacity, plus "
    #     "cycle count and peak performance capability — which together tell you "
    #     "whether degradation or a faulty charging circuit is the real cause. ..."
    # ),
}

# SPECIFIC_IDENTIFIERS: keywords that indicate the user has already named a
# specific model/version in their question. If any of these appear, the
# clarification turn is skipped (no point asking "which model?" if they said it).
# ⚠️ DOMAIN-SPECIFIC — replace with identifiers from your domain.
SPECIFIC_IDENTIFIERS = [
    # ── Examples for consumer devices ──────────
    # "model 12 pro", "series x2", "gen 3", "2023 edition",
    # ── Examples for industrial machinery ────────────────────
    # "model 3000", "series x2", "revision c",
]

# When a question lacks a specific identifier, ask this clarification.
# ⚠️ DOMAIN-SPECIFIC — write a question and a realistic example answer.
CLARIFICATION_QUESTION = "Which specific model and software version are you working with?"
CLARIFICATION_ANSWER   = "Model X, software version 2.3."

# =============================================================
# 1) HELPERS
# =============================================================
def extract_steps(output_text: str) -> list:
    """Pull numbered steps from an existing answer."""
    lines = output_text.split("\n")
    return [l.strip() for l in lines if l.strip().startswith("Step ")]

# =============================================================
# 2) CONVERSATION TEMPLATES
# =============================================================
# Each function takes one row from the SFT dataset and returns a
# ShareGPT-format conversation dict, or None if the template cannot
# be applied to that row (e.g. answer has too few steps).
#
# The five patterns themselves are domain-agnostic — only the
# map contents (PREVENTION_MAP, FOLLOWUP_MAP, identifiers) are domain-specific.


def make_already_tried_conversation(row: dict):
    """User says they already tried the first step; model skips it."""
    steps = extract_steps(row["output"])
    if len(steps) < 2:
        return None

    first_action = steps[0].replace("Step 1:", "").replace("Step 1 :", "").strip()
    if len(first_action) > 80:
        first_action = first_action[:80] + "..."

    continuation = (
        f"Understood — since {first_action.lower()} didn't resolve it, "
        f"let's move to the next diagnostic step.\n\n"
        + "\n".join(steps[1:])
    ) if steps[1:] else (
        "In that case, the issue is likely more complex and will require "
        "further investigation by a qualified technician."
    )

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": row["input"]},
            {"from": "gpt",    "value": row["output"]},
            {"from": "human",  "value": f"I already tried {first_action} but it didn't help. What else can I do?"},
            {"from": "gpt",    "value": continuation},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "already_tried",
    }


def make_clarification_conversation(row: dict):
    """Model asks for the specific model/version before giving full answer."""
    question = row["input"]
    if any(kw in question.lower() for kw in SPECIFIC_IDENTIFIERS):
        return None  # question already names a specific model — skip

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": question},
            {"from": "gpt",    "value": f"I can help with that. Before I give you specific steps — {CLARIFICATION_QUESTION}"},
            {"from": "human",  "value": CLARIFICATION_ANSWER},
            {"from": "gpt",    "value": row["output"]},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "clarification",
    }


def make_escalation_conversation(row: dict):
    """User confirms initial steps didn't work; model escalates to next level."""
    steps = extract_steps(row["output"])
    if len(steps) < 4:
        return None  # need enough steps to have an escalation tier

    software_steps = steps[:3]
    hardware_steps = steps[3:]
    if not hardware_steps:
        return None

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": row["input"]},
            {"from": "gpt",    "value": row["output"]},
            {"from": "human",  "value": "I tried all the initial steps you mentioned but the problem is still there."},
            {"from": "gpt",    "value": (
                "Since the initial steps didn't resolve it, this points to a deeper issue. "
                "Here is what to check next:\n\n"
                + "\n".join(hardware_steps)
                + "\n\nIf these steps also don't resolve the issue, professional inspection "
                "is recommended — the component may need physical replacement or specialist diagnosis."
            )},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "escalation",
    }


def make_success_confirmation_conversation(row: dict):
    """User confirms fix worked; model gives domain-specific prevention advice."""
    topic      = row.get("topic", "")
    prevention = PREVENTION_MAP.get(topic) or PREVENTION_MAP.get(None, (
        "Follow the manufacturer's recommended maintenance schedule and keep "
        "all firmware and software updated to prevent recurrence."
    ))

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": row["input"]},
            {"from": "gpt",    "value": row["output"]},
            {"from": "human",  "value": "That worked! Thank you. Is there anything I can do to prevent this from happening again?"},
            {"from": "gpt",    "value": (
                f"Great to hear it's resolved! Here's how to prevent this in the future:\n\n"
                f"{prevention}\n\n"
                f"If the issue returns despite these precautions, it may indicate a deeper "
                f"underlying fault — don't ignore recurring symptoms."
            )},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "success_prevention",
    }


def make_stepbystep_conversation(row: dict):
    """User asks to be guided one step at a time; model paces them."""
    steps = extract_steps(row["output"])
    if len(steps) < 3:
        return None

    total = len(steps)
    s1 = steps[0].replace("Step 1:", "").replace("Step 1 :", "").strip()
    s2 = steps[1].replace("Step 2:", "").replace("Step 2 :", "").strip()
    s3 = steps[2].replace("Step 3:", "").replace("Step 3 :", "").strip()

    s1 = s1[:120] + ("..." if len(s1) > 120 else "")
    s2 = s2[:120] + ("..." if len(s2) > 120 else "")
    s3 = s3[:120] + ("..." if len(s3) > 120 else "")

    remaining = f"Only {total - 3} more steps after this." if total > 3 else "This is the last step — almost done!"

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": row["input"]},
            {"from": "gpt",    "value": (
                f"{row['output']}\n\n"
                f"I can guide you through this one step at a time if you'd like. "
                f"Just say 'guide me' and I'll walk you through each step."
            )},
            {"from": "human",  "value": "Yes please guide me step by step"},
            {"from": "gpt",    "value": (
                f"Let's go through this together. There are {total} steps total.\n\n"
                f"**Step 1 of {total}:** {s1}\n\n"
                f"Take your time. Let me know when you're done and I'll give you the next step."
            )},
            {"from": "human",  "value": "Done, what's next?"},
            {"from": "gpt",    "value": (
                f"Great work on Step 1.\n\n"
                f"**Step 2 of {total}:** {s2}\n\n"
                f"Let me know when you're ready to continue."
            )},
            {"from": "human",  "value": "I'm having trouble with this step, I can't seem to do it"},
            {"from": "gpt",    "value": (
                f"No problem — let's work through Step 2 before moving on.\n\n"
                f"A few things to check:\n"
                f"- Make sure you have the correct tools or access required for this step\n"
                f"- Reread the step carefully — sometimes a small detail is easy to miss\n"
                f"- If there is resistance or an unexpected state, stop and describe what you see\n\n"
                f"What specifically are you stuck on? I'll guide you through it."
            )},
            {"from": "human",  "value": "Ok I figured it out, ready for next step"},
            {"from": "gpt",    "value": (
                f"Well done for working through it!\n\n"
                f"**Step 3 of {total}:** {s3}\n\n"
                f"{remaining} Let me know when you're ready to continue."
            )},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "stepbystep_guidance",
    }


def make_followup_question_conversation(row: dict):
    """User asks a conceptual follow-up about something in the answer."""
    topic = row.get("topic", "")
    if topic not in FOLLOWUP_MAP:
        return None  # no follow-up defined for this topic — skip

    follow_q, follow_a = FOLLOWUP_MAP[topic]

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": row["input"]},
            {"from": "gpt",    "value": row["output"]},
            {"from": "human",  "value": follow_q},
            {"from": "gpt",    "value": follow_a},
        ],
        "topic":      row.get("topic", ""),
        "source":     row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "turn_type":  "followup_question",
    }

# =============================================================
# 3) MAIN
# =============================================================
def main():
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"SFT dataset not found: {DATASET_PATH}\n"
            f"Run generate_dataset.py (Stage 0) first, or set SFT_DATASET_PATH."
        )

    print("\n" + "=" * 62)
    print("  STAGE 2a — MULTI-TURN CONVERSATION GENERATOR")
    print("=" * 62)

    data = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"\n✅ Loaded {len(data)} base entries from {DATASET_PATH}")

    # (name, generator function, probability of applying per row)
    # Probabilities < 1.0 introduce natural variety — not every entry
    # gets every template, which prevents the multi-turn dataset from
    # being too formulaic.
    generators = [
        ("already_tried",       make_already_tried_conversation,       0.4),
        ("clarification",       make_clarification_conversation,        0.3),
        ("escalation",          make_escalation_conversation,           0.35),
        ("success_prevention",  make_success_confirmation_conversation, 0.4),
        ("followup_question",   make_followup_question_conversation,    0.3),
        ("stepbystep_guidance", make_stepbystep_conversation,           0.35),
    ]

    multiturn_entries = []
    for row in data:
        for _name, fn, prob in generators:
            if random.random() < prob:
                result = fn(row)
                if result is not None:
                    multiturn_entries.append(result)

    print(f"✅ Generated {len(multiturn_entries)} multi-turn conversations")
    print(f"   Expansion ratio: {len(multiturn_entries) / max(1, len(data)):.1f}× base dataset size")

    turn_counts: dict = {}
    for e in multiturn_entries:
        t = e.get("turn_type", "?")
        turn_counts[t] = turn_counts.get(t, 0) + 1

    print("\n  By turn type:")
    for t, n in sorted(turn_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<30} {n}")

    if turn_counts.get("followup_question", 0) == 0:
        print(
            "\n  ⚠️  No followup_question turns generated — FOLLOWUP_MAP is empty.\n"
            "     Add entries to FOLLOWUP_MAP in this file to enable this pattern."
        )

    with open(MULTITURN_OUT_PATH, "w", encoding="utf-8") as f:
        for e in multiturn_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n💾 Saved → {MULTITURN_OUT_PATH}")
    print(f"\n   Next: python multiturn_train.py")


if __name__ == "__main__":
    main()
