"""
generate_dataset.py — Stage 0 of the SLM fine-tuning pipeline.

Reads PDF manuals/documentation, chunks the text, calls an LLM to generate
troubleshooting Q&A pairs, validates and deduplicates them, then writes
training_dataset.jsonl — the exact file SLM_finetune_v2.py (Stage 1) expects.

Pipeline position:
    generate_dataset.py  ← YOU ARE HERE
          │
          ▼  training_dataset.jsonl
    SLM_finetune_v2.py   (Stage 1 — single-turn SFT)
          │
          ▼
    multiturn_train.py   (Stage 2 — multi-turn SFT)
          │
          ▼
    mistral_dpo_train.py (Stage 3 — DPO preference training)

Required environment variables (never hardcode in source):
    OPENAI_API_KEY   — OpenAI write key (or swap call_llm() for another provider)
    PDF_DIR          — directory containing your PDF files (default: ./pdfs)
    SAMPLES_PER_PDF  — target samples per PDF file (default: 150)
    BUDGET_USD       — maximum spend in USD (default: 5.00)
    OUTPUT_FILE      — output path (default: training_dataset.jsonl)

⚠️ DOMAIN-SPECIFIC — the following sections must be rewritten for your domain:
    1. DOMAIN_RULES  — commands, required terms, banned terms, scenario templates
    2. TOPIC_KEYWORDS — keywords for diversity tracking
    3. extract_error_codes() — regex for your domain's error/fault code format
    4. SYSTEM_PROMPT — must match the prompt used in all training stages
"""

import os
import json
import random
import re
import hashlib
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher

from tqdm import tqdm
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# 0) CONFIG  ← edit environment variables, not this block
# =============================================================

# ── Spend / budget ───────────────────────────────────────────
BUDGET_USD        = float(os.environ.get("BUDGET_USD", "5.00"))
SAFETY_BUFFER     = 0.10                           # keep 10% as reserve
EFFECTIVE_BUDGET  = BUDGET_USD * (1 - SAFETY_BUFFER)

# Prices for gpt-4o-mini (update if you switch models)
PRICE_INPUT_PER_TOK  = 0.150 / 1_000_000
PRICE_OUTPUT_PER_TOK = 0.600 / 1_000_000

# ── Input / output ───────────────────────────────────────────
PDF_DIR          = Path(os.environ.get("PDF_DIR", "./pdfs"))
SAMPLES_PER_PDF  = int(os.environ.get("SAMPLES_PER_PDF", "150"))
OUTPUT_FILE      = os.environ.get("OUTPUT_FILE", "training_dataset.jsonl")
REJECTED_FILE    = "rejected.jsonl"
CHECKPOINT_FILE  = "checkpoint.json"
SPEND_FILE       = "spend_tracker.json"

# ── Chunking ─────────────────────────────────────────────────
CHUNK_WORDS     = 380
OVERLAP_WORDS   = 60
MIN_CHUNK_WORDS = 120
MAX_PER_CHUNK   = 2    # samples generated per LLM call (2 = fewer API calls)

# =============================================================
# ⚠️ DOMAIN-SPECIFIC — rewrite this entire block for your domain
# =============================================================
#
# DOMAIN_RULES controls what makes a valid sample for each document.
# Each PDF file gets its own entry, keyed by the PDF filename stem
# (e.g. "service_manual" for "service_manual.pdf").
#
# Required fields per entry:
#   allowed_commands  — commands, tools, menu paths that appear in your docs.
#                       At least one must appear in every generated output.
#   required_terms    — domain vocabulary; 2+ must appear in every output.
#   banned_terms      — terms from OTHER domains (prevents cross-contamination).
#   input_scenarios   — template strings for generating diverse inputs.
#                       Use {placeholders} — they are filled with random values
#                       from PLACEHOLDER_VALUES below.
#
# The example below is for smartphone hardware repair.
# Replace with your own domain's vocabulary, tools, and scenario patterns.
#
DOMAIN_RULES = {
    # ── Example: one entry per PDF file stem ─────────────────
    # "your_manual": {
    #     "allowed_commands": ["Your Command 1", "Your Tool 2"],
    #     "required_terms":   ["Term A", "Term B"],
    #     "banned_terms":     ["UnrelatedTerm"],
    #     "input_scenarios":  ["error {code} during {operation}"],
    # },

    # ── Smartphone repair example — replace or extend ───
    "service_manual": {
        "allowed_commands": [
            "Safe Mode", "Recovery Mode", "Factory Reset", "ADB Logcat",
            "Battery Health Check", "Display Test", "Sensor Diagnostics",
            "Firmware Reflash", "Cache Partition Wipe",
            "Charging Port Inspection", "Multimeter Continuity Test",
        ],
        "required_terms": [
            "battery", "display", "charging port", "firmware",
            "logic board", "flex cable", "sensor", "digitizer",
        ],
        "banned_terms": ["ECU", "CAN bus", "crankshaft", "BIOS", "POST"],
        "input_scenarios": [
            "error {code} shown on {subsystem} after {event}",
            "{subsystem} unresponsive after {event}",
            "error {code} returns immediately after a factory reset of {subsystem}",
            "{component} not detected by the device after {event}",
            "diagnostic test {block} reports an implausible value for {component}",
            "{component} works intermittently and drops out under load",
            "device fails to complete {operation} and reboots",
            "{subsystem} calibration required after {component} replacement",
            "battery drains rapidly with no rogue process visible in logs",
            "device not charging on channel {channel} of the test rig",
        ],
    },
}

# Placeholder values used to fill {placeholders} in input_scenarios above.
# ⚠️ DOMAIN-SPECIFIC — replace with values relevant to your domain.
PLACEHOLDER_VALUES = {
    "code":      ["E-101", "E-204", "E-330", "E-415", "E-502", "E-617"],
    "subsystem": ["display", "battery", "charging", "camera", "audio", "cellular radio"],
    "event":     ["a drop", "a firmware update", "component replacement", "water ingress"],
    "channel":   ["01", "05", "12", "23"],
    "component": ["digitizer", "battery flex cable", "charging port", "proximity sensor"],
    "block":     ["001", "003", "010", "032"],
    "byte":      ["0", "1", "2", "3"],
    "operation": ["a firmware reflash", "display calibration", "a cache partition wipe"],
}

# Keywords used to detect topic for diversity tracking.
# ⚠️ DOMAIN-SPECIFIC — add keywords that cover the major topics in your manuals.
TOPIC_KEYWORDS_DEFAULT = [
    "display", "battery", "charging", "camera", "audio", "cellular",
    "sensor", "firmware", "calibration", "digitizer", "storage", "wifi",
]

# ⚠️ DOMAIN-SPECIFIC — this prompt must be IDENTICAL to the SYSTEM_PROMPT
# in SLM_finetune_v2.py, multiturn_train.py, and mistral_dpo_train.py,
# AND to whatever prompt you use at inference time.
SYSTEM_PROMPT = (
    "You are an expert smartphone hardware repair technician with 10+ years of experience. "
    "STRICT RULE: Always exhaust ALL software and diagnostic checks before suggesting "
    "any physical disassembly or component replacement. "
    "Your response structure must ALWAYS follow this order: "
    "(1) Identify the likely root cause based on the fault code and symptoms. "
    "(2) Read and record all related error codes and device diagnostic logs. "
    "(3) Check relevant sensor readings and live device telemetry. "
    "(4) Run built-in diagnostic tests using service tools. "
    "(5) Physical inspection only after software diagnostics are exhausted. "
    "(6) Component replacement ONLY as a last resort with supporting evidence. "
    "Never replace a part until diagnostic data points conclusively to it. "
    "If the technician has already completed a step, acknowledge it and give the NEXT step."
)

# =============================================================
# LOGGING
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("generation.log")],
)
log = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =============================================================
# 1) SPEND TRACKER
# =============================================================
class SpendTracker:
    def __init__(self, budget):
        self.budget = budget
        self.data = {"total_spent": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}
        if Path(SPEND_FILE).exists():
            with open(SPEND_FILE) as f:
                self.data = json.load(f)
        log.info("Budget $%.4f | Spent $%.4f | Remaining $%.4f",
                 budget, self.data["total_spent"], self.remaining)

    @property
    def remaining(self):
        return self.budget - self.data["total_spent"]

    def will_exceed(self, est_in, est_out):
        estimated = (est_in * PRICE_INPUT_PER_TOK + est_out * PRICE_OUTPUT_PER_TOK) * 1.2
        return self.remaining < estimated

    def record(self, in_tok, out_tok):
        cost = in_tok * PRICE_INPUT_PER_TOK + out_tok * PRICE_OUTPUT_PER_TOK
        self.data["total_spent"] += cost
        self.data["calls"]       += 1
        self.data["tokens_in"]   += in_tok
        self.data["tokens_out"]  += out_tok
        with open(SPEND_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def report(self):
        d = self.data
        log.info(
            "SPEND: $%.5f / $%.2f | %d calls | in=%d out=%d tokens | $%.5f remaining",
            d["total_spent"], self.budget, d["calls"],
            d["tokens_in"], d["tokens_out"], self.remaining,
        )


spend = SpendTracker(EFFECTIVE_BUDGET)

# =============================================================
# 2) CHECKPOINT
# =============================================================
def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done), f)

def chunk_id(source, chunk):
    return hashlib.sha1(f"{source}:{chunk[:200]}".encode()).hexdigest()

# =============================================================
# 3) PDF EXTRACTION
# =============================================================
def extract_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        log.error("PDF extraction failed for %s: %s", path, e)
        return ""

def extract_error_codes(text: str) -> list:
    """
    ⚠️ DOMAIN-SPECIFIC — replace this regex with your domain's fault/error code format.
    Examples:
      Generic E-codes : r'\bE-[0-9]{3}\b'
      IBM             : r'\\b(?:DSI|BNH|FLB)\\d+[A-Z]?\\b'
      Oracle          : r'\\bORA-\\d{4,5}\\b'
    """
    codes = re.findall(r'\bE-[0-9]{3}\b', text)   # example: E-101 style service codes
    return sorted(set(codes))[:30]

# =============================================================
# 4) CHUNKING — procedure-boundary aware
# =============================================================
def create_chunks(text: str) -> list:
    """
    Splits text into overlapping word-count chunks.
    Avoids breaking inside numbered procedure steps.
    """
    sections = re.split(r'\n(?=[A-Z][A-Z\s]{4,}\n)', text)
    chunks = []
    for sec in sections:
        sentences = re.split(r'(?<=[.!?])\s+', sec)
        current, cur_len, in_proc = [], 0, False
        for sent in sentences:
            words = sent.split()
            wlen  = len(words)
            if re.match(r'^(?:\d+\.|Step \d+[:\.])\s', sent.strip()):
                in_proc = True
            if not sent.strip():
                in_proc = False
            if cur_len + wlen > CHUNK_WORDS and cur_len >= MIN_CHUNK_WORDS and not in_proc:
                chunks.append(" ".join(current))
                current = current[-OVERLAP_WORDS:] + words
                cur_len = len(current)
                in_proc = False
            else:
                current.extend(words)
                cur_len += wlen
        if cur_len >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(current))
    return chunks

# Reference/appendix chunks (dense code tables with no procedure text) are
# skipped — they contain no actionable troubleshooting steps for the LLM to use.
_CODE_DENSE = re.compile(r'\b[A-Z]{2,4}\d{3,6}[A-Z]?\b', re.MULTILINE)
_PROC_SIG   = re.compile(r'(?:Step \d|procedure|to resolve|solution|perform)', re.I)

def is_reference_chunk(chunk: str) -> bool:
    if len(chunk.split()) < MIN_CHUNK_WORDS:
        return True
    codes = len(_CODE_DENSE.findall(chunk))
    if codes >= 4 and not _PROC_SIG.search(chunk):
        return True
    lines      = [l.strip() for l in chunk.split('\n') if l.strip()]
    code_lines = sum(1 for l in lines if re.match(r'^[A-Z]{2,4}\d{3,}', l))
    return bool(lines) and code_lines / len(lines) > 0.40

# =============================================================
# 5) TOPIC DETECTION — for diversity tracking
# =============================================================
def topic_of(text: str, keywords: list) -> str:
    tl = text.lower()
    for kw in keywords:
        if kw in tl:
            return kw
    return "other"

# =============================================================
# 6) FUZZY COMMAND MATCH
# =============================================================
def cmd_in(text: str, cmd: str, threshold=0.82) -> bool:
    """Returns True if cmd appears in text, allowing for minor formatting differences."""
    tl, cl = text.lower(), cmd.lower().strip()
    if cl in tl:
        return True
    cn = re.sub(r'\s+', ' ', cl)
    tn = re.sub(r'\s+', ' ', tl)
    w    = len(cn) + int(len(cn) * 0.25)
    step = max(1, len(cn) // 5)
    for i in range(0, max(1, len(tn) - w + 1), step):
        if SequenceMatcher(None, cn, tn[i:i+w]).ratio() >= threshold:
            return True
    return False

# =============================================================
# 7) PLACEHOLDER FILL — randomises scenario templates
# =============================================================
def fill_scenario(template: str) -> str:
    """Replace {placeholder} tokens with random values from PLACEHOLDER_VALUES."""
    def replace(m):
        key = m.group(1)
        opts = PLACEHOLDER_VALUES.get(key, [key])
        return random.choice(opts)
    return re.sub(r'\{(\w+)\}', replace, template)

# =============================================================
# 8) PROMPT BUILDER
# =============================================================
def build_prompt(chunk: str, rules: dict, error_codes: list, used_inputs: list) -> str:
    cmds      = "\n".join(f"  - {c}" for c in rules["allowed_commands"])
    terms     = ", ".join(rules["required_terms"])
    scenarios = "\n".join(
        f"  - {fill_scenario(s)}"
        for s in random.sample(rules["input_scenarios"], min(5, len(rules["input_scenarios"])))
    )
    codes_str = ", ".join(error_codes[:15]) if error_codes else "see document"
    recent    = "\n".join(f"  - {inp}" for inp in used_inputs[-6:]) if used_inputs else "  (none yet)"

    return f"""Return ONLY valid JSON. No markdown. No explanation.

Generate EXACTLY 2 troubleshooting samples in this format:
{{
  "instruction": "Diagnose and fix the issue",
  "input": "<specific enterprise issue>",
  "output": "Issue:\\n<one sentence>\\n\\nPossible Causes:\\n1. <technical cause with component name>\\n2. <technical cause with component name>\\n\\nSolution:\\nStep 1: <command or diagnostic step from document>\\nStep 2: <command or diagnostic step from document>\\nStep 3: <command or diagnostic step from document>\\nStep 4: <command or diagnostic step from document>\\n\\nExpected Outcome:\\n<specific observable result>"
}}

━━ STRICT OUTPUT RULES ━━
- Output MUST have all 4 sections: Issue, Possible Causes, Solution, Expected Outcome
- Solution MUST have AT LEAST 4 steps (Step 1 through Step 4 minimum)
- Each step MUST reference a specific command, tool, or procedure from the ALLOWED list
- Output MUST be at least 160 words
- Each cause MUST name a specific component, module, or configuration parameter

━━ BANNED PHRASES (will be rejected) ━━
Do NOT use ANY of these:
  - "misconfiguration" → say WHICH parameter is wrong and WHERE
  - "network issue" → name the specific protocol, bus, or link
  - "verify the configuration" → name WHICH config and WHICH parameter
  - "check the logs" → name WHICH log and WHICH message pattern
  - "restart the service" → only use if the document names the restart procedure
  - "contact support" → never use this
  - "unknown error" → always name the specific fault code or condition

━━ DO NOT REPEAT THESE RECENTLY USED INPUTS ━━
{recent}

━━ USE THESE SCENARIO PATTERNS INSTEAD ━━
{scenarios}

━━ FAULT / ERROR CODES FROM THIS DOCUMENT ━━
{codes_str}

━━ ALLOWED COMMANDS / TOOLS / PROCEDURES ━━
{cmds}

━━ REQUIRED DOMAIN TERMS (2+ must appear in output) ━━
{terms}

━━ DOCUMENT CHUNK ━━
{chunk}"""

# =============================================================
# 9) LLM CALL — gpt-4o-mini with retry + budget guard
# =============================================================
def call_llm(prompt: str):
    est_in, est_out = len(prompt.split()) * 4 // 3, 700
    if spend.will_exceed(est_in, est_out):
        log.warning("Budget ceiling reached — stopping generation")
        return None, 0, 0
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            in_tok  = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            spend.record(in_tok, out_tok)
            return resp.choices[0].message.content.strip(), in_tok, out_tok
        except Exception as e:
            wait = 2 ** attempt
            log.warning("API error (attempt %d): %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    return None, 0, 0

# =============================================================
# 10) JSON PARSER — handles raw and markdown-wrapped JSON
# =============================================================
def parse_json(raw: str) -> list:
    raw = raw.replace("```json", "").replace("```", "").strip()
    objects, stack, start = [], [], None
    for i, ch in enumerate(raw):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                try:
                    objects.append(json.loads(raw[start:i + 1]))
                except Exception:
                    pass
    return objects

# =============================================================
# 11) VALIDATION
# =============================================================
REQUIRED_SECTIONS = ["Issue:", "Possible Causes:", "Solution:", "Expected Outcome:"]

GENERIC_PHRASES = [
    "network issue", "misconfiguration", "system issue", "unknown error",
    "check the logs", "restart the service", "contact support",
    "verify the configuration", "check for errors", "review the documentation",
    "check the configuration", "ensure proper configuration",
    "configuration issue", "connectivity issue", "general misconfiguration",
]

def log_rejection(sample: dict, source: str, reason: str):
    with open(REJECTED_FILE, "a") as f:
        f.write(json.dumps({
            "source": source,
            "reason": reason,
            "input":  str(sample.get("input", ""))[:100],
        }) + "\n")

def validate(sample: dict, rules: dict, full_doc: str, source: str):
    if not isinstance(sample, dict):
        return False, "not_dict"

    for key in ("instruction", "input", "output"):
        if key not in sample or not isinstance(sample[key], str):
            return False, f"missing_{key}"

    inp, out = sample["input"].strip(), sample["output"]
    out_lower = out.lower()

    if len(inp) < 15:
        return False, "input_too_short"

    for sec in REQUIRED_SECTIONS:
        if sec not in out:
            return False, f"missing_{sec.replace(':', '')}"

    steps = len(re.findall(r"Step \d+:", out))
    if steps < 4:
        return False, f"only_{steps}_steps_need_4"

    if len(out.split()) < 140:
        return False, "output_too_short"

    for phrase in GENERIC_PHRASES:
        if phrase in out_lower:
            return False, f"generic:'{phrase}'"

    # At least one allowed command must appear in the output
    if not any(cmd_in(out, c) for c in rules["allowed_commands"]):
        return False, "no_allowed_command"

    # That command must also exist in the source document (not hallucinated)
    has_grounded = any(
        cmd_in(out, c) and cmd_in(full_doc, c)
        for c in rules["allowed_commands"]
    )
    if not has_grounded:
        return False, "ungrounded_command"

    term_hits = sum(1 for t in rules["required_terms"] if t.lower() in out_lower)
    if term_hits < 2:
        return False, f"only_{term_hits}_domain_terms"

    for banned in rules["banned_terms"]:
        if banned.lower() in out_lower:
            return False, f"banned:{banned}"

    # Cross-domain contamination check
    for other_name, other_rules in DOMAIN_RULES.items():
        if other_name == source:
            continue
        for cmd in other_rules["allowed_commands"]:
            if len(cmd) > 6 and cmd.lower() in out_lower:
                return False, f"cross_domain:{other_name}"

    return True, ""

# =============================================================
# 12) DEDUPLICATION — exact prefix + keyword fingerprint
# =============================================================
class ThematicDeduper:
    """
    Two-level deduplication:
    1. Exact normalised prefix — catches near-identical phrasings.
    2. Keyword fingerprint   — catches thematic repeats (same fault code +
       same command = same sample rephrased differently).
    """
    def __init__(self):
        self.input_prefixes  = set()
        self.output_prefixes = set()
        self.keyword_sigs    = set()

    def _keyword_sig(self, text: str):
        words = re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', text)
        return frozenset(sorted(set(words))[:6])

    def is_duplicate(self, inp: str, out: str) -> bool:
        ip = re.sub(r'\s+', ' ', inp.lower().strip())[:90]
        op = re.sub(r'\s+', ' ', out.lower().strip())[:90]
        if ip in self.input_prefixes:
            return True
        if op in self.output_prefixes:
            return True
        ksig = self._keyword_sig(inp + " " + out[:200])
        if len(ksig) >= 3 and ksig in self.keyword_sigs:
            return True
        self.input_prefixes.add(ip)
        self.output_prefixes.add(op)
        self.keyword_sigs.add(ksig)
        return False

# =============================================================
# 13) DIFFICULTY SCORING
# =============================================================
def score_difficulty(sample: dict, commands: list) -> str:
    out   = sample.get("output", "")
    words = len(out.split())
    steps = len(re.findall(r"Step \d+:", out))
    cmds  = sum(1 for c in commands if cmd_in(out, c))
    score = (words >= 300) + (words >= 500) + (steps >= 5) + (steps >= 7) + (cmds >= 3) + (cmds >= 5)
    return ["easy", "easy", "medium", "medium", "hard", "hard", "hard"][min(score, 6)]

# =============================================================
# 14) COUNT EXISTING OUTPUT (for resume)
# =============================================================
def count_existing() -> dict:
    counts: dict = {}
    if not Path(OUTPUT_FILE).exists():
        return counts
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                src = json.loads(line).get("source", "")
                counts[src] = counts.get(src, 0) + 1
            except Exception:
                pass
    return counts

# =============================================================
# 15) MAIN
# =============================================================
def main():
    # ── Discover PDFs ────────────────────────────────────────
    if not PDF_DIR.exists():
        log.error("PDF_DIR '%s' does not exist — set PDF_DIR env var or create the directory", PDF_DIR)
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error("No PDF files found in %s", PDF_DIR)
        return

    log.info("Found %d PDF file(s): %s", len(pdf_files), [p.name for p in pdf_files])

    done_hashes = load_checkpoint()
    deduper     = ThematicDeduper()
    existing    = count_existing()

    # Pre-load deduper from existing output to avoid re-generating known samples
    used_inputs_by_source: dict = defaultdict(list)
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    deduper.is_duplicate(obj.get("input", ""), obj.get("output", ""))
                    src = obj.get("source", "")
                    used_inputs_by_source[src].append(obj.get("input", ""))
                except Exception:
                    pass

    log.info("Existing samples: %s", existing)
    log.info("Target per PDF  : %d", SAMPLES_PER_PDF)

    rejection_counts = Counter()
    total_new        = 0

    for pdf_path in pdf_files:
        source  = pdf_path.stem          # e.g. "service_manual" for "service_manual.pdf"
        already = existing.get(source, 0)
        needed  = SAMPLES_PER_PDF - already

        if needed <= 0:
            log.info("[%s] Already complete (%d samples)", source, already)
            continue

        # Use domain-specific rules if defined, otherwise use a generic fallback
        if source in DOMAIN_RULES:
            rules = DOMAIN_RULES[source]
        else:
            log.warning(
                "[%s] No entry in DOMAIN_RULES — using generic fallback. "
                "Add a '%s' entry for better quality.", source, source
            )
            # Generic fallback pulls allowed_commands / required_terms from
            # all defined domains combined — coarse but functional
            all_cmds  = [c for r in DOMAIN_RULES.values() for c in r["allowed_commands"]]
            all_terms = [t for r in DOMAIN_RULES.values() for t in r["required_terms"]]
            all_scens = [s for r in DOMAIN_RULES.values() for s in r["input_scenarios"]]
            rules = {
                "allowed_commands": all_cmds,
                "required_terms":   all_terms,
                "banned_terms":     [],
                "input_scenarios":  all_scens,
            }

        topic_keywords = TOPIC_KEYWORDS_DEFAULT

        log.info("[%s] Extracting text from %s ...", source, pdf_path)
        text = extract_text(pdf_path)
        if not text:
            log.error("[%s] No text extracted — skipping", source)
            continue

        error_codes = extract_error_codes(text)
        log.info("[%s] Found %d fault codes: %s...", source, len(error_codes), error_codes[:5])

        chunks_raw = create_chunks(text)
        chunks     = [c for c in chunks_raw if not is_reference_chunk(c)]
        log.info("[%s] Chunks: %d total → %d after reference filter", source, len(chunks_raw), len(chunks))

        random.shuffle(chunks)

        MAX_PER_TOPIC = max(3, needed // max(1, len(topic_keywords)))
        topic_counts  = Counter()
        count         = 0
        used_inputs   = used_inputs_by_source[source]

        for chunk in tqdm(chunks, desc=f"[{source}]"):
            if count >= needed:
                break
            if spend.remaining <= 0:
                log.warning("Budget exhausted — stopping")
                break

            cid = chunk_id(source, chunk)
            if cid in done_hashes:
                continue

            topic = topic_of(chunk, topic_keywords)
            if topic_counts[topic] >= MAX_PER_TOPIC:
                done_hashes.add(cid)
                continue

            prompt = build_prompt(chunk, rules, error_codes, used_inputs)
            raw, _, _ = call_llm(prompt)
            done_hashes.add(cid)
            save_checkpoint(done_hashes)

            if raw is None:
                break

            accepted_here = 0
            for obj in parse_json(raw):
                valid, reason = validate(obj, rules, text, source)
                if not valid:
                    log_rejection(obj, source, reason)
                    rejection_counts[reason] += 1
                    continue

                if deduper.is_duplicate(obj["input"], obj["output"]):
                    log_rejection(obj, source, "duplicate")
                    rejection_counts["duplicate"] += 1
                    continue

                obj["source"]     = source
                obj["topic"]      = topic
                obj["difficulty"] = score_difficulty(obj, rules["allowed_commands"])

                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                used_inputs.append(obj["input"])
                topic_counts[topic] += 1
                count     += 1
                total_new += 1
                accepted_here += 1

                log.info(
                    "[%s] +1 (%d/%d) $%.5f left | diff=%-6s | %s",
                    source, already + count, SAMPLES_PER_PDF,
                    spend.remaining, obj["difficulty"], obj["input"][:70],
                )

                if accepted_here >= MAX_PER_CHUNK:
                    break

    # ── Final reports ─────────────────────────────────────────
    spend.report()

    total_rejected = sum(rejection_counts.values())
    total_attempts = total_new + total_rejected
    if total_attempts:
        rate = total_rejected / total_attempts
        log.info("Rejection rate: %.1f%% (%d rejected / %d attempted)", rate * 100, total_rejected, total_attempts)

    if rejection_counts:
        log.info("=== REJECTION BREAKDOWN ===")
        for reason, cnt in rejection_counts.most_common(15):
            log.info("  %-45s %d", reason, cnt)

    final = count_existing()
    log.info("=== FINAL SAMPLE COUNTS ===")
    for src, n in final.items():
        log.info("  %-30s %d", src, n)
    log.info("New samples this run : %d", total_new)
    log.info("Output file          : %s", OUTPUT_FILE)
    log.info("Ready for Stage 1    : python SLM_finetune_v2.py")


if __name__ == "__main__":
    main()
