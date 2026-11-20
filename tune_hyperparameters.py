"""
tune_hyperparameters.py — Hyperparameter search for Stage 1 SFT.

Finds the best learning rate, batch size, LoRA rank, and regularisation
settings for YOUR dataset before committing to a full training run.

Library  : Optuna  (pip install optuna)
Method   : TPE (Tree-structured Parzen Estimator) — Bayesian optimisation.
           Learns which parts of the search space are promising from each
           trial result, rather than sampling randomly.
Pruner   : MedianPruner — kills a trial early if its eval loss after K
           steps is worse than the median of all trials so far at the
           same step. Saves GPU time on clearly bad configs.

Typical use:
    python tune_hyperparameters.py          # 10 trials, ~2–4 hrs on T4
    python tune_hyperparameters.py          # re-run: resumes from study DB

Output:
    best_hyperparams.json     — paste these into SLM_finetune_v2.py
    optuna_study.db           — SQLite study (survives crashes; resume-able)

Pipeline position:
    tune_hyperparameters.py    ← YOU ARE HERE (optional, run before Stage 1)
          │
          ▼  best_hyperparams.json
    SLM_finetune_v2.py         (Stage 1 — single-turn SFT)

Required environment variables:
    SFT_DATASET_PATH — Alpaca JSONL dataset (default: training_dataset.jsonl)

Optional:
    N_TRIALS        — number of Optuna trials (default: 10)
    TRIAL_EPOCHS    — epochs per trial (default: 1 — enough to distinguish)
    STUDY_DB        — SQLite path for resume (default: optuna_study.db)
"""

import torch
assert torch.cuda.is_available(), "Enable GPU runtime"

import os
import json
import math
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
DATASET_PATH  = os.environ.get("SFT_DATASET_PATH", "training_dataset.jsonl")
N_TRIALS      = int(os.environ.get("N_TRIALS",     "10"))
TRIAL_EPOCHS  = int(os.environ.get("TRIAL_EPOCHS", "1"))
STUDY_DB      = os.environ.get("STUDY_DB",         "optuna_study.db")
OUTPUT_FILE   = "best_hyperparams.json"
MAX_SEQ_LEN   = 2048   # shorter than full training → faster per trial
SEED          = 3407

# ⚠️ DOMAIN-SPECIFIC — same as SLM_finetune_v2.py
SYSTEM_PROMPT = (
    "You are an expert technical support specialist with 10+ years of experience "
    "in your domain. You provide detailed, step-by-step diagnostic and repair "
    "guidance. Your responses always: (1) identify the likely root cause, "
    "(2) provide numbered diagnostic steps from least invasive to most invasive, "
    "(3) note when professional service is required, and (4) cite the source of your "
    "information. You never guess — you reason systematically."
)

# ── Imports ───────────────────────────────────────────────────────────
print("Loading dependencies...")

import transformers.utils as _tu
if not hasattr(_tu, "is_bitsandbytes_multi_backend_available"):
    _tu.is_bitsandbytes_multi_backend_available = lambda: False

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners  import MedianPruner
except ImportError:
    raise ImportError("Install Optuna first:  pip install optuna")

from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Transformers 4.56.x bug fix
try:
    import transformers.utils.hub as _hub_mod
    import transformers.tokenization_utils_base as _tok_mod
    _orig = _hub_mod.list_repo_templates
    def _safe(repo_id, *a, **kw):
        try: return _orig(repo_id, *a, **kw)
        except Exception: return []
    _hub_mod.list_repo_templates = _safe
    if hasattr(_tok_mod, "list_repo_templates"):
        _tok_mod.list_repo_templates = _safe
except Exception:
    pass


# ── Search space ──────────────────────────────────────────────────────
#
# TPE works best with 5–20 parameters. These are the ones that matter
# most for Mistral 7B + LoRA on small domain datasets:
#
#   learning_rate         — most impactful; log-uniform to cover 10× range
#   lora_r                — LoRA rank: higher = more capacity, slower training
#   lora_alpha            — LoRA scaling (typically 1–2× r)
#   lora_dropout          — regularisation inside the adapter
#   warmup_ratio          — fraction of steps used for LR warm-up
#   per_device_batch_size — constrained by T4 VRAM; 1 or 2 for 7B 4-bit
#   gradient_accumulation — effective batch = bs × grad_acc; tune together
#
def suggest_params(trial: optuna.Trial) -> dict:
    lr        = trial.suggest_float("learning_rate",  1e-5, 5e-4, log=True)
    lora_r    = trial.suggest_categorical("lora_r",   [8, 16, 32])
    # alpha is often set to 2×r, but letting Optuna choose the ratio is useful
    alpha_mul = trial.suggest_categorical("alpha_multiplier", [1, 2])
    dropout   = trial.suggest_float("lora_dropout",   0.0, 0.15)
    warmup    = trial.suggest_float("warmup_ratio",   0.05, 0.20)
    bs        = trial.suggest_categorical("per_device_batch_size", [1, 2])
    grad_acc  = trial.suggest_categorical("gradient_accumulation_steps", [4, 8])

    return {
        "learning_rate":                lr,
        "lora_r":                       lora_r,
        "lora_alpha":                   lora_r * alpha_mul,
        "lora_dropout":                 dropout,
        "warmup_ratio":                 warmup,
        "per_device_train_batch_size":  bs,
        "gradient_accumulation_steps":  grad_acc,
    }


# ── Dataset + format ──────────────────────────────────────────────────
def load_and_format_dataset():
    """Load and format the dataset once; reuse across all trials."""
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            f"Run generate_dataset.py (Stage 0) first."
        )

    raw = load_dataset("json", data_files=DATASET_PATH, split="train")
    raw = raw.shuffle(seed=SEED)
    split = raw.train_test_split(test_size=0.15, seed=SEED)

    def fmt(batch):
        texts = []
        for inst, inp, out in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            user = f"{inst}\n\n{inp}".strip() if inp else inst
            texts.append(
                f"<s>[INST] {SYSTEM_PROMPT}\n\n{user} [/INST] {out}</s>"
            )
        return {"text": texts}

    keep = ["instruction", "input", "output"]
    train_ds = split["train"].select_columns(keep).map(fmt, batched=True, remove_columns=keep)
    eval_ds  = split["test"].select_columns(keep).map(fmt,  batched=True, remove_columns=keep)
    return train_ds, eval_ds


# ── Objective function ────────────────────────────────────────────────
def make_objective(train_ds, eval_ds):
    """
    Returns an Optuna objective that:
      1. Loads a fresh Mistral 7B base (4-bit)
      2. Applies LoRA with the trial's suggested rank/alpha/dropout
      3. Trains for TRIAL_EPOCHS
      4. Returns eval_loss (lower = better)
      5. Reports intermediate loss to Optuna for pruning
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} — suggested params:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print('='*60)

        # Load base model fresh each trial (Unsloth caches weights; overhead is small)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=params["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        n_train      = len(train_ds)
        steps_epoch  = math.ceil(
            n_train / (params["per_device_train_batch_size"]
                       * params["gradient_accumulation_steps"])
        )
        total_steps  = steps_epoch * TRIAL_EPOCHS
        eval_steps   = max(steps_epoch // 3, 5)   # evaluate ~3× per epoch

        # Optuna pruning callback: report eval loss after each eval step
        class PruningCallback(torch.nn.Module):
            """Reports intermediate eval_loss to Optuna so bad trials are killed early."""
            def __init__(self, trial):
                super().__init__()
                self._trial = trial

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    self._trial.report(metrics["eval_loss"], step=state.global_step)
                    if self._trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Pruned at step {state.global_step} "
                            f"(eval_loss={metrics['eval_loss']:.4f})"
                        )

        from transformers import TrainerCallback
        class _PruningCB(TrainerCallback):
            def __init__(self, trial): self._trial = trial
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    self._trial.report(metrics["eval_loss"], step=state.global_step)
                    if self._trial.should_prune():
                        raise optuna.TrialPruned()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            packing=True,
            callbacks=[_PruningCB(trial)],
            args=SFTConfig(
                per_device_train_batch_size=params["per_device_train_batch_size"],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=params["gradient_accumulation_steps"],
                num_train_epochs=TRIAL_EPOCHS,
                learning_rate=params["learning_rate"],
                warmup_ratio=params["warmup_ratio"],
                optim="adamw_8bit",
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="no",       # don't save checkpoints during search
                seed=SEED,
                output_dir=f"tune_trial_{trial.number}",
                report_to="none",
                max_seq_length=MAX_SEQ_LEN,
            ),
        )

        try:
            trainer.train()
        except optuna.TrialPruned:
            raise

        metrics   = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", float("inf"))
        print(f"\n  Trial {trial.number} eval_loss = {eval_loss:.4f}")

        # Free VRAM before next trial
        del model, trainer
        torch.cuda.empty_cache()

        return eval_loss

    return objective


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  HYPERPARAMETER SEARCH — Mistral 7B + LoRA")
    print(f"  Sampler : TPE (Bayesian)")
    print(f"  Pruner  : MedianPruner")
    print(f"  Trials  : {N_TRIALS}")
    print(f"  Epochs  : {TRIAL_EPOCHS} per trial")
    print(f"  Study DB: {STUDY_DB}  (resume-able)")
    print("="*60)

    train_ds, eval_ds = load_and_format_dataset()
    print(f"\n✅ Dataset: {len(train_ds)} train / {len(eval_ds)} eval samples")

    study = optuna.create_study(
        study_name="mistral_lora_tuning",
        direction="minimize",             # minimise eval_loss
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(
            n_startup_trials=3,           # don't prune until 3 trials are done
            n_warmup_steps=10,            # don't prune before 10 eval steps
        ),
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,             # resume an interrupted search
    )

    already_done = len(study.trials)
    remaining    = max(0, N_TRIALS - already_done)
    if remaining == 0:
        print(f"\n✅ Study already has {already_done} completed trials — loading best.")
    else:
        print(f"\n✅ Starting {remaining} trial(s) ({already_done} already completed)...")
        study.optimize(
            make_objective(train_ds, eval_ds),
            n_trials=remaining,
            catch=(RuntimeError,),   # catch OOM without crashing the whole study
        )

    # ── Results ───────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "="*60)
    print("  BEST TRIAL")
    print("="*60)
    print(f"  Trial number : #{best.number}")
    print(f"  eval_loss    : {best.value:.4f}  (perplexity {math.exp(best.value):.2f})")
    print("\n  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Reconstruct lora_alpha (stored separately from params)
    alpha_mul   = best.params.get("alpha_multiplier", 2)
    lora_r      = best.params["lora_r"]
    best_config = {
        "learning_rate":               best.params["learning_rate"],
        "lora_r":                      lora_r,
        "lora_alpha":                  lora_r * alpha_mul,
        "lora_dropout":                best.params["lora_dropout"],
        "warmup_ratio":                best.params["warmup_ratio"],
        "per_device_train_batch_size": best.params["per_device_train_batch_size"],
        "gradient_accumulation_steps": best.params["gradient_accumulation_steps"],
        "_eval_loss":                  best.value,
        "_perplexity":                 math.exp(best.value),
        "_trial_number":               best.number,
        "_note": (
            "Generated by tune_hyperparameters.py. "
            "These are 1-epoch estimates — you may want to increase LR slightly "
            "for full multi-epoch runs, as the optimal LR tends to scale with training length."
        ),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n💾 Best config saved → {OUTPUT_FILE}")
    print("\n  Paste these into SLM_finetune_v2.py:")
    print(f"    LR            = {best_config['learning_rate']:.2e}")
    print(f"    LORA_R        = {best_config['lora_r']}")
    print(f"    LORA_ALPHA    = {best_config['lora_alpha']}")
    print(f"    LORA_DROPOUT  = {best_config['lora_dropout']:.3f}")
    print(f"    WARMUP_RATIO  = {best_config['warmup_ratio']:.3f}")
    print(f"    PER_DEVICE_BS = {best_config['per_device_train_batch_size']}")
    print(f"    GRAD_ACC      = {best_config['gradient_accumulation_steps']}")

    # ── Trial summary table ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  ALL TRIALS — sorted by eval_loss")
    print("="*60)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(completed, key=lambda t: t.value):
        ppl = math.exp(t.value)
        mark = " ← BEST" if t.number == best.number else ""
        print(f"  #{t.number:>2}  loss={t.value:.4f}  ppl={ppl:.2f}"
              f"  lr={t.params['learning_rate']:.1e}"
              f"  r={t.params['lora_r']}"
              f"  bs={t.params['per_device_train_batch_size']}{mark}")

    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        print(f"\n  {len(pruned)} trial(s) were pruned early by MedianPruner.")

    print(f"\n✅ Done. Next: update SLM_finetune_v2.py with the values above,")
    print(f"   then run the full pipeline from Stage 1.")


if __name__ == "__main__":
    main()
