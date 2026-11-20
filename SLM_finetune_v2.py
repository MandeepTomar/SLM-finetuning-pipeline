import torch
assert torch.cuda.is_available(), "Enable GPU runtime (Colab → Runtime → GPU)"

# =========================
# 1) IMPORTS
# =========================
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

import os
import json
import math

# Stub for missing transformers symbol (sentence-transformers can downgrade transformers)
import transformers.utils as _tu
if not hasattr(_tu, "is_bitsandbytes_multi_backend_available"):
    _tu.is_bitsandbytes_multi_backend_available = lambda: False

# Transformers 4.56.x bug fix: list_repo_templates() raises on repos with no
# additional_chat_templates folder (all unsloth quantized models).
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


# =========================
# 2) CONFIG
# =========================
@dataclass
class FineTuneConfig:

    # ---------------- MODEL ----------------
    model_name: str
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    dtype: Optional[str] = None

    # ---------------- DATASET ----------------
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_path: Optional[str] = None
    split: str = "train"
    seed: int = 3407

    # ---------------- TRAINING ----------------
    output_dir: str = "outputs"
    lora_save_path: str = "lora_adapters"

    per_device_bs: int = 2
    grad_acc_steps: int = 4         # effective batch size = 8

    # ── V2 TUNING NOTES ──────────────────────────────────────
    # Dataset grew from 800 → 854 entries but more importantly
    # quality improved: longer chain-of-thought outputs,
    # differential diagnosis structure, and source citations.
    # Longer average output length (~2100 chars vs old ~2082)
    # with more structured reasoning.
    #
    # Key changes from v1:
    #  - epochs: 3 (was 4) — larger dataset, richer outputs,
    #    overfitting risk is lower but best checkpoint saved anyway
    #  - lr: 3e-4 (was 4.97e-4) — slightly lower for stability
    #    with longer sequences
    #  - lora_r: 16 (was 8) — richer reasoning in outputs
    #    needs more capacity to capture diagnostic patterns
    #  - lora_alpha: 32 (was 64) — standard 2× ratio, more
    #    conservative; avoids over-scaling with larger r
    #  - lora_dropout: 0.05 (was 0.07)
    #  - warmup_ratio: 0.08 (was 0.06) — slightly longer warmup
    #    for stability with richer data
    # ─────────────────────────────────────────────────────────
    epochs: int = 3
    lr: float = 3e-4
    warmup_ratio: float = 0.08

    logging_steps: int = 10
    eval_steps: int = 96            # ~1 epoch: 769 train / 8 eff_bs ≈ 96 steps
    save_steps: int = 96

    packing: bool = True

    # ---------------- LORA ----------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    target_modules: List[str] = None

    use_gc: bool = True

    # ---------------- SAVE ----------------
    save_merged: bool = False
    merged_save_path: str = "merged_fp16_model"

    # ---------------- HUGGINGFACE HUB ----------------
    # The trained adapters are pushed to the HuggingFace account that owns
    # HUB_TOKEN. To save models on YOUR organisation's account:
    #   1. Create a HF account/org for your team (huggingface.co)
    #   2. Generate a WRITE token: Settings → Access Tokens → New token (write)
    #   3. Export it before running:  export HF_TOKEN=hf_xxxx
    #   4. Set hub_model_id to "<your-org-or-username>/<model-name>"
    # Never commit the token to source control.
    push_to_hub: bool = False
    hub_model_id: str = ""           # e.g. "your-org/mistral-domain-sft-v2"
    hub_token: Optional[str] = None  # read from HF_TOKEN env var (see run config below)

    # ---------------- SYSTEM PROMPT ----------------
    # We inject a system prompt to teach the model its role during training.
    # ⚠️ DOMAIN-SPECIFIC — REPLACE FOR YOUR USE CASE.
    # Write a prompt describing YOUR domain expert (e.g. a hardware repair
    # technician for device service manuals). Keep the same structural rules:
    # root cause → numbered steps (least to most invasive) → escalation → citations.
    # The SAME prompt must be used at inference time in your system.
    use_system_prompt: bool = True
    system_prompt: str = (
        "You are an expert technical support specialist with 10+ years of experience "
        "in your domain. You provide detailed, step-by-step diagnostic and repair "
        "guidance. Your responses always: (1) identify the likely root cause, "
        "(2) provide numbered diagnostic steps from least invasive to most invasive, "
        "(3) note when professional service is required, and (4) cite the source of your "
        "information. You never guess — you reason systematically."
    )


# =========================
# 3) FINETUNER
# =========================
class UnslothFineTuner:

    def __init__(self, cfg: FineTuneConfig):

        self.cfg = cfg

        if self.cfg.target_modules is None:
            self.cfg.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        self.model = None
        self.tokenizer = None

    # =========================================================
    # LOAD MODEL
    # =========================================================
    def load_model(self):

        print(f"\n✅ Loading model: {self.cfg.model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            dtype=self.cfg.dtype,
            load_in_4bit=self.cfg.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cfg.lora_r,
            target_modules=self.cfg.target_modules,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.cfg.use_gc,
            random_state=self.cfg.seed,
        )

        self.model.print_trainable_parameters()

        print("✅ PEFT Model:", isinstance(self.model, PeftModel))
        print("✅ Device:", next(self.model.parameters()).device)
        print("✅ Dtype :", next(self.model.parameters()).dtype)

    # =========================================================
    # LOAD DATASET
    # =========================================================
    def load_dataset(self):

        if self.cfg.dataset_path:

            print(f"\n✅ Loading local dataset: {self.cfg.dataset_path}")
            path = self.cfg.dataset_path

            if path.endswith(".jsonl") or path.endswith(".json"):
                full_ds = load_dataset("json", data_files=path, split="train")
            elif path.endswith(".csv"):
                full_ds = load_dataset("csv", data_files=path, split="train")
            else:
                raise ValueError(f"❌ Unsupported file type: {path}")

        else:
            print(f"\n✅ Loading dataset: {self.cfg.dataset_name}")
            try:
                full_ds = load_dataset(self.cfg.dataset_name, split=self.cfg.split)
            except Exception:
                full_ds = load_dataset(path=self.cfg.dataset_name, split=self.cfg.split)

        split_ds = full_ds.train_test_split(test_size=0.10, seed=self.cfg.seed)
        train_ds = split_ds["train"].shuffle(seed=self.cfg.seed)
        eval_ds  = split_ds["test"].shuffle(seed=self.cfg.seed)

        print("\n✅ DATASET SPLIT")
        print(f"Train Samples : {len(train_ds)}")
        print(f"Eval Samples  : {len(eval_ds)}")

        eval_ds.to_json("unseen_test_dataset.jsonl")
        print("💾 Saved unseen evaluation dataset → unseen_test_dataset.jsonl")

        return train_ds, eval_ds

    # =========================================================
    # PROMPT TEMPLATES
    # =========================================================
    def _eos(self):
        return self.tokenizer.eos_token or ""

    def _mistral_prompt(self):
        """
        Mistral instruct format: <s>[INST] {system}\n\n{user} [/INST] {assistant}</s>
        System prompt is prepended to the first user turn (no separate system block).
        """
        if self.cfg.use_system_prompt:
            return "<s>[INST] {system}\n\n{instruction}\n\n{input} [/INST] {output}</s>"
        else:
            return "<s>[INST] {instruction}\n\n{input} [/INST] {output}</s>"

    # =========================================================
    # FORMAT MISTRAL (Mistral instruct format)
    # =========================================================
    def _format_alpaca(self, batch):
        """Mistral instruct format for Alpaca-style data."""
        texts = []
        inp_list = batch.get("input", [""] * len(batch["instruction"]))

        for ins, inp, out in zip(batch["instruction"], inp_list, batch["output"]):
            user_block = (ins or "").strip()
            if inp and inp.strip():
                user_block += "\n\n" + inp.strip()
            if self.cfg.use_system_prompt:
                text = f"<s>[INST] {self.cfg.system_prompt}\n\n{user_block} [/INST] {out or ''}</s>"
            else:
                text = f"<s>[INST] {user_block} [/INST] {out or ''}</s>"
            texts.append(text)

        return {"text": texts}

    def _format_dolly(self, batch):
        ctx_list = batch.get("context", [""] * len(batch["instruction"]))
        texts = []
        for ins, ctx, resp in zip(batch["instruction"], ctx_list, batch["response"]):
            user_block = (ins or "").strip()
            if ctx and ctx.strip():
                user_block += "\n\n" + ctx.strip()
            if self.cfg.use_system_prompt:
                texts.append(f"<s>[INST] {self.cfg.system_prompt}\n\n{user_block} [/INST] {resp or ''}</s>")
            else:
                texts.append(f"<s>[INST] {user_block} [/INST] {resp or ''}</s>")
        return {"text": texts}

    def _format_sharegpt(self, batch):
        texts = []
        conv_key = "conversations" if "conversations" in batch else "messages"
        for conv in batch[conv_key]:
            if conv is None:
                texts.append("")
                continue
            turns = []
            system_text = ""
            for t in conv:
                if isinstance(t, dict) and "from" in t and "value" in t:
                    role, value = t["from"], t["value"]
                elif isinstance(t, dict) and "role" in t and "content" in t:
                    role, value = t["role"], t["content"]
                else:
                    continue
                if role == "system":
                    system_text = value
                else:
                    turns.append((role, value))

            chat = ""
            first_user = True
            i = 0
            while i < len(turns):
                role, content = turns[i]
                if role in ["human", "user"]:
                    if first_user and system_text:
                        chat += f"<s>[INST] {system_text}\n\n{content} [/INST]"
                        first_user = False
                    else:
                        chat += f"[INST] {content} [/INST]"
                    # pair with next assistant turn
                    if i + 1 < len(turns) and turns[i + 1][0] in ["gpt", "assistant"]:
                        chat += f" {turns[i + 1][1]}</s>"
                        i += 2
                        continue
                i += 1
            texts.append(chat.strip())
        return {"text": texts}

    def _format_text(self, batch):
        EOS = self._eos()
        if "text" not in batch:
            raise ValueError("Expected text column.")
        return {"text": [(t or "") + EOS for t in batch["text"]]}

    # =========================================================
    # FORMAT DATASET
    # =========================================================
    def format_dataset(self, ds):

        print("\n✅ Inferring dataset format...")
        cols = set(ds.column_names)

        if self.cfg.dataset_path:
            if {"instruction", "output"}.issubset(cols):
                print("✅ Format: ALPACA (with system prompt)" if self.cfg.use_system_prompt else "✅ Format: ALPACA")
                keep = [c for c in ["instruction", "input", "output"] if c in cols]
                ds_slim = ds.select_columns(keep)
                return ds_slim.map(self._format_alpaca, batched=True, remove_columns=ds_slim.column_names)

            if "text" in cols:
                print("✅ Format: TEXT")
                return ds.map(self._format_text, batched=True, remove_columns=ds.column_names)

            raise ValueError(f"❌ Unknown local dataset format: {sorted(list(cols))}")

        name = self.cfg.dataset_name
        if name == "tatsu-lab/alpaca":
            return ds.map(self._format_alpaca, batched=True, remove_columns=ds.column_names)
        if name == "databricks/databricks-dolly-15k":
            return ds.map(self._format_dolly, batched=True, remove_columns=ds.column_names)
        if "conversations" in cols or "messages" in cols:
            return ds.map(self._format_sharegpt, batched=True, remove_columns=ds.column_names)
        if "text" in cols:
            return ds.map(self._format_text, batched=True, remove_columns=ds.column_names)
        if {"instruction", "output"}.issubset(cols):
            return ds.map(self._format_alpaca, batched=True, remove_columns=ds.column_names)

        raise ValueError(f"❌ Could not infer dataset format. Columns: {sorted(list(cols))}")

    # =========================================================
    # TRAIN
    # =========================================================
    def train(self, train_ds, eval_ds):

        print("\n✅ Starting training (V2)...")
        print(f"   model        = {self.cfg.model_name}")
        print(f"   lr           = {self.cfg.lr:.2e}")
        print(f"   lora_r       = {self.cfg.lora_r}")
        print(f"   lora_alpha   = {self.cfg.lora_alpha}")
        print(f"   lora_dropout = {self.cfg.lora_dropout}")
        print(f"   epochs       = {self.cfg.epochs}")
        print(f"   warmup_ratio = {self.cfg.warmup_ratio}")
        print(f"   system_prompt= {'YES' if self.cfg.use_system_prompt else 'NO'}")

        trainer = SFTTrainer(

            model=self.model,
            tokenizer=self.tokenizer,

            train_dataset=train_ds,
            eval_dataset=eval_ds,

            dataset_text_field="text",
            packing=self.cfg.packing,

            args=SFTConfig(

                per_device_train_batch_size=self.cfg.per_device_bs,
                per_device_eval_batch_size=self.cfg.per_device_bs,
                gradient_accumulation_steps=self.cfg.grad_acc_steps,

                num_train_epochs=self.cfg.epochs,
                learning_rate=self.cfg.lr,
                warmup_ratio=self.cfg.warmup_ratio,

                optim="adamw_8bit",

                logging_steps=self.cfg.logging_steps,

                eval_strategy="steps",
                eval_steps=self.cfg.eval_steps,

                save_strategy="steps",
                save_steps=self.cfg.save_steps,

                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,

                seed=self.cfg.seed,
                output_dir=self.cfg.output_dir,
                report_to="none",
            ),
        )

        trainer.train()

        # ─────────────────────────────────────────────────────
        # FINAL EVALUATION
        # Remove the notebook progress callback first — it loses
        # state after train() and crashes on standalone evaluate()
        # ─────────────────────────────────────────────────────
        try:
            from transformers.utils.notebook import NotebookProgressCallback
            trainer.remove_callback(NotebookProgressCallback)
        except Exception:
            pass

        print("\n✅ Running final evaluation...")
        metrics = trainer.evaluate()

        print("\n📊 FINAL METRICS")
        print("=" * 40)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        if "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            print(f"\n🔥 Perplexity: {perplexity:.4f}")
            print(f"\n📌 Target for ChatGPT-comparable quality: Perplexity < 1.8")
            print(f"   (V1 baseline was 2.04 — lower is better)")

        return trainer

    # =========================================================
    # SAVE
    # =========================================================
    def save(self):

        print(f"\n✅ Saving LoRA adapters → {self.cfg.lora_save_path}")
        self.model.save_pretrained(self.cfg.lora_save_path)
        self.tokenizer.save_pretrained(self.cfg.lora_save_path)

        if self.cfg.save_merged:
            print(f"\n✅ Merging + Saving full model → {self.cfg.merged_save_path}")
            merged = self.model.merge_and_unload()
            merged.save_pretrained(self.cfg.merged_save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(self.cfg.merged_save_path)

    # =========================================================
    # INFERENCE HELPER (V2 NEW — for testing after training)
    # =========================================================
    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        """
        Run inference with the trained model using the same prompt
        format as training (system prompt + instruction + input).
        """
        FastLanguageModel.for_inference(self.model)

        system = self.cfg.system_prompt if self.cfg.use_system_prompt else ""
        if system:
            prompt = (
                f"### System:\n{system}\n\n"
                f"### Instruction:\nDiagnose and fix the issue\n\n"
                f"### Input:\n{question}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task, paired with an input "
                f"that provides further context.\n\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\nDiagnose and fix the issue\n\n"
                f"### Input:\n{question}\n\n"
                f"### Response:\n"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,       # low temp for factual repair advice
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response:" in decoded:
            return decoded.split("### Response:")[-1].strip()
        return decoded

    # =========================================================
    # PUSH TO HUB
    # =========================================================
    def push_to_hub(self):
        if not self.cfg.push_to_hub or not self.cfg.hub_model_id:
            return

        from huggingface_hub import login
        if self.cfg.hub_token:
            login(token=self.cfg.hub_token)

        print(f"\n✅ Pushing LoRA adapters → {self.cfg.hub_model_id}")
        self.model.push_to_hub(self.cfg.hub_model_id)
        self.tokenizer.push_to_hub(self.cfg.hub_model_id)
        print(f"✅ Done — https://huggingface.co/{self.cfg.hub_model_id}")

    # =========================================================
    # RUN
    # =========================================================
    def run(self):

        self.load_model()

        train_raw, eval_raw = self.load_dataset()

        train_ds = self.format_dataset(train_raw)
        eval_ds  = self.format_dataset(eval_raw)

        print("\n✅ Sample formatted training text:\n")
        print(train_ds["text"][0][:1500])
        print("...")

        self.train(train_ds, eval_ds)
        self.save()
        self.push_to_hub()

        # ─────────────────────────────────────────────────────
        # QUICK SMOKE TEST
        # ─────────────────────────────────────────────────────
        print("\n🧪 Running smoke test inference...")
        # Replace with a question from YOUR domain to sanity-check the model
        test_q = os.environ.get(
            "SMOKE_TEST_QUESTION",
            "Describe the diagnostic steps for an intermittent hardware fault.",
        )
        print(f"Q: {test_q}\n")
        answer = self.generate(test_q, max_new_tokens=400)
        print(f"A: {answer[:800]}...")

        print("\n🔥 TRAINING COMPLETE (V2)")


# =====================================================================
# 4) RUN — TEAM INTEGRATION GUIDE
# ─────────────────────────────────────────────────────────────────────
# DATASET REQUIREMENTS (what your JSONL must look like):
#   One JSON object per line with fields:
#     instruction : the task/question phrasing
#     input       : additional context (symptom description, excerpt) — optional
#     output      : the ideal expert answer
#   Quality guidelines that produced the best results in our runs:
#     - Root-cause analysis section in every output
#     - Numbered steps ordered least → most invasive
#     - Explicit "when to escalate to professional service" guidance
#     - Source citations (manual page/section numbers)
#   Recommended size: 800+ entries; balance easy/medium/hard cases.
#
# HYPERPARAMETERS (tuned for ~850-entry datasets on a T4/L4 GPU):
#   epochs=3, lr=2e-4..3e-4, lora_r=16, lora_alpha=32 (2× r),
#   lora_dropout=0.05, warmup_ratio=0.08.
#   Larger datasets (5k+): reduce epochs to 1–2 and consider lr=1e-4.
#
# COMPLEXITY:
#   Time  : O(epochs × N × L) transformer forward/backward passes, where
#           N = dataset rows and L = max_seq_length (attention is O(L²)
#           per layer — halving max_seq_length ~4×'s attention speed).
#   Space : 4-bit base model (~4 GB VRAM for 7B) + LoRA adapters
#           (only ~0.5% of params trainable) + activations O(bs × L).
#           Fits a 16 GB T4 at bs=2, grad_acc=4, seq_len=4096.
# =====================================================================

def main():
    """Entry point — edit the config below for your training run."""
    cfg = FineTuneConfig(

        # ── Model ─────────────────────────────────────────────
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",

        # ── Dataset ───────────────────────────────────────────
        # Point this at YOUR training dataset (JSONL with
        # instruction / input / output fields — Alpaca format).
        # Generate it from your PDF corpus first.
        dataset_path=os.environ.get("SFT_DATASET_PATH", "/content/training_dataset.jsonl"),

        # ── Outputs ───────────────────────────────────────────
        lora_save_path="mistral_v1_lora",
        output_dir="mistral_v1_outputs",

        # ── Training ──────────────────────────────────────────
        epochs=3,
        per_device_bs=2,
        grad_acc_steps=4,            # effective batch size = 8
        lr=2e-4,
        warmup_ratio=0.08,

        # ── LoRA ──────────────────────────────────────────────
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,

        # ── System Prompt ─────────────────────────────────────
        use_system_prompt=True,
        # system_prompt is set to the default persona in FineTuneConfig —
        # replace it there with your domain expert persona

        # ── Misc ──────────────────────────────────────────────
        max_seq_length=4096,
        use_gc=True,
        save_merged=False,           # set True if you want to export full model

        # ── HuggingFace Hub ───────────────────────────────────
        # Models are pushed to the account that owns HF_TOKEN.
        # Set both env vars before running:
        #   export HF_TOKEN=hf_xxxx                      (write token from YOUR account)
        #   export HUB_MODEL_ID=your-org/mistral-sft-v2  (repo to create/update)
        push_to_hub=bool(os.environ.get("HF_TOKEN")),
        hub_model_id=os.environ.get("HUB_MODEL_ID", ""),
        hub_token=os.environ.get("HF_TOKEN"),
    )

    trainer = UnslothFineTuner(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

# =========================
# RELOADING IN A NEW SESSION
# =========================
# Once pushed, load your model from HuggingFace like this:
#
#   from unsloth import FastLanguageModel
#   from peft import PeftModel
#
#   base, tokenizer = FastLanguageModel.from_pretrained(
#       "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#       max_seq_length=2048,
#       load_in_4bit=True,
#   )
#   model = PeftModel.from_pretrained(base, "your-org/mistral-sft-v2")
#   FastLanguageModel.for_inference(model)
#
# Then pass model + tokenizer into your own inference system,
# using the same Mistral instruct prompt format as training:
#   <s>[INST] {SYSTEM_PROMPT}\n\n{user question} [/INST]
# =========================
