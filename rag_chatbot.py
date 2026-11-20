"""
rag_chatbot.py — Stage 4: RAG inference with your fine-tuned SLM.

Loads your PDF manuals into a vector index, then runs an interactive
chat loop where every user question is answered by:
  1. Retrieving the most relevant chunks from your PDFs (RAG)
  2. Injecting those chunks into the prompt as context
  3. Generating a response with your fine-tuned Mistral adapter

This means the model answers from YOUR documents, not from its
pre-training knowledge — grounded, citation-ready responses.

Pipeline position:
    mistral_dpo_train.py   (Stage 3b — DPO training)
          │
          ▼  final LoRA adapter
    rag_chatbot.py           ← YOU ARE HERE  (Stage 4 — inference)

GPU recommended for fast generation (works on CPU but slow).

Required environment variables:
    MODEL_PATH   — HF repo or local path of your fine-tuned adapter
                   (default: mistral_dpo_lora)
    PDF_DIR      — directory containing your PDF files
                   (default: ./pdfs)

Optional:
    TOP_K        — number of chunks to retrieve per query (default: 3)
    CHUNK_WORDS  — target words per chunk (default: 300)
    INDEX_CACHE  — path to save/load the FAISS index so PDFs aren't
                   re-embedded on every run (default: ./rag_index_cache)

⚠️ DOMAIN-SPECIFIC:
    SYSTEM_PROMPT — must be IDENTICAL to the one used during training.
    Copy from generate_dataset.py without changes.
"""

import os
import json
import re
import pickle
import hashlib
from pathlib import Path

# =============================================================
# 0) CONFIG
# =============================================================
MODEL_PATH   = os.environ.get("MODEL_PATH",   "mistral_dpo_lora")
PDF_DIR      = Path(os.environ.get("PDF_DIR", "./pdfs"))
TOP_K        = int(os.environ.get("TOP_K",    "3"))
CHUNK_WORDS  = int(os.environ.get("CHUNK_WORDS", "300"))
INDEX_CACHE  = Path(os.environ.get("INDEX_CACHE", "./rag_index_cache"))

# ⚠️ DOMAIN-SPECIFIC — must be IDENTICAL to all training scripts
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
print("Loading dependencies...")

from pypdf import PdfReader

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "RAG dependencies missing. Install them:\n"
        "  pip install faiss-cpu sentence-transformers\n"
        "  (use faiss-gpu instead of faiss-cpu if you have a GPU)"
    )

import torch

# =============================================================
# 2) PDF LOADING + CHUNKING
# =============================================================
def load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return ""


def chunk_text(text: str, source: str) -> list:
    """Split text into overlapping word-count chunks with source metadata."""
    words    = text.split()
    overlap  = CHUNK_WORDS // 5
    chunks   = []
    i        = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_WORDS]
        if len(chunk_words) >= 50:          # skip tiny trailing fragments
            chunks.append({
                "text":   " ".join(chunk_words),
                "source": source,
                "offset": i,
            })
        i += CHUNK_WORDS - overlap
    return chunks


def load_documents(pdf_dir: Path) -> list:
    """Load all PDFs in pdf_dir and return a flat list of chunk dicts."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}\n"
            f"Set PDF_DIR env var or place PDFs in {pdf_dir}"
        )

    print(f"\n📄 Loading {len(pdf_files)} PDF(s)...")
    all_chunks = []
    for pdf_path in pdf_files:
        text = load_pdf(pdf_path)
        if not text.strip():
            continue
        chunks = chunk_text(text, pdf_path.stem)
        all_chunks.extend(chunks)
        print(f"   {pdf_path.name:<40} {len(chunks)} chunks")

    print(f"   Total chunks: {len(all_chunks)}")
    return all_chunks

# =============================================================
# 3) VECTOR INDEX — build or load from cache
# =============================================================
def _corpus_hash(chunks: list) -> str:
    """Fingerprint the corpus so we know when to rebuild the index."""
    raw = json.dumps([c["text"][:50] for c in chunks], ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_or_load_index(chunks: list, embedder: SentenceTransformer):
    """
    Build a FAISS flat-L2 index from chunk embeddings.
    Caches to INDEX_CACHE so re-runs are instant.
    """
    INDEX_CACHE.mkdir(parents=True, exist_ok=True)
    h          = _corpus_hash(chunks)
    index_path = INDEX_CACHE / f"index_{h}.faiss"
    meta_path  = INDEX_CACHE / f"meta_{h}.pkl"

    if index_path.exists() and meta_path.exists():
        print(f"\n✅ Loading cached index ({h})...")
        index  = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"   {index.ntotal} vectors loaded")
        return index, meta

    print(f"\n✅ Building vector index ({len(chunks)} chunks)...")
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)      # inner product on normalised = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"   Index built: {index.ntotal} vectors, dim={dim}")
    print(f"   Cached → {INDEX_CACHE}/")
    return index, chunks


def retrieve(query: str, index, chunks: list, embedder: SentenceTransformer) -> list:
    """Return the TOP_K most relevant chunks for a query."""
    q_emb = embedder.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    scores, ids = index.search(q_emb, TOP_K)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results

# =============================================================
# 4) MODEL LOADING
# =============================================================
def load_model():
    """Load the fine-tuned Mistral adapter for inference."""
    # Check for local adapters first (same session), then fall back to MODEL_PATH
    local_candidates = ["mistral_dpo_lora", "mistral_multiturn_lora", "mistral_v1_lora"]
    load_path = MODEL_PATH
    for candidate in local_candidates:
        if os.path.exists(candidate):
            load_path = candidate
            break

    print(f"\n✅ Loading fine-tuned model: {load_path}")

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("   ⚠️  No GPU detected — running on CPU (generation will be slow)")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=use_gpu,       # 4-bit only works on CUDA
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model from '{load_path}': {e}\n\n"
            f"Make sure you have run the training pipeline first, or set:\n"
            f"  export MODEL_PATH=your-org/mistral-dpo"
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model ready")
    return model, tokenizer

# =============================================================
# 5) PROMPT BUILDER
# =============================================================
def build_rag_prompt(history: list, context_chunks: list) -> str:
    """
    Build a Mistral instruct prompt with:
      - retrieved document context injected after the system prompt
      - full conversation history for multi-turn coherence

    Format:
      <s>[INST] {SYSTEM_PROMPT}

      RELEVANT DOCUMENTATION:
      [Source: ...] chunk text...

      {first user message} [/INST] {first assistant response}</s>
      [INST] {second user message} [/INST] {second response}</s>
      ...
      [INST] {latest user message} [/INST]
    """
    context_block = ""
    if context_chunks:
        parts = []
        for c in context_chunks:
            parts.append(f"[Source: {c['source']}]\n{c['text']}")
        context_block = "\nRELEVANT DOCUMENTATION:\n" + "\n\n".join(parts) + "\n"

    prompt = ""
    for i, (role, text) in enumerate(history):
        if role == "user":
            if i == 0:
                prompt += f"<s>[INST] {SYSTEM_PROMPT}{context_block}\n{text} [/INST]"
            else:
                prompt += f"[INST] {text} [/INST]"
        else:
            prompt += f" {text}</s>"

    return prompt


def generate_response(prompt: str, model, tokenizer) -> str:
    """Run inference and return the generated text."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# =============================================================
# 6) CHAT LOOP
# =============================================================
def print_sources(chunks: list):
    """Print retrieved source chunks compactly."""
    print(f"\n  📎 Retrieved {len(chunks)} source chunk(s):")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{i}] {c['source']} (score {c['score']:.3f}): {preview}...")


def chat_loop(model, tokenizer, index, chunks, embedder):
    print("\n" + "=" * 62)
    print("  RAG CHATBOT — powered by your fine-tuned SLM")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  PDFs   : {PDF_DIR}")
    print(f"  Top-K  : {TOP_K} chunks per query")
    print("=" * 62)
    print("\n  Commands:")
    print("    Type your question and press Enter")
    print("    'sources' — toggle showing retrieved chunks")
    print("    'clear'   — start a new conversation")
    print("    'quit'    — exit\n")

    history      = []
    show_sources = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "clear":
            history = []
            print("  [Conversation cleared]\n")
            continue

        if user_input.lower() == "sources":
            show_sources = not show_sources
            print(f"  [Source display {'ON' if show_sources else 'OFF'}]\n")
            continue

        # Retrieve relevant chunks for this turn
        retrieved = retrieve(user_input, index, chunks, embedder)
        if show_sources:
            print_sources(retrieved)

        # Build prompt with history + retrieved context
        history.append(("user", user_input))
        prompt   = build_rag_prompt(history, retrieved)
        response = generate_response(prompt, model, tokenizer)
        history.append(("assistant", response))

        print(f"\nAssistant: {response}\n")

# =============================================================
# 7) MAIN
# =============================================================
def main():
    print("\n" + "=" * 62)
    print("  STAGE 4 — RAG CHATBOT")
    print("=" * 62)

    # ── Load documents ────────────────────────────────────────
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR}\n"
            f"Set PDF_DIR env var or create the directory and add your PDFs."
        )

    all_chunks = load_documents(PDF_DIR)

    # ── Build embedding model + index ────────────────────────
    print("\n✅ Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    index, chunks = build_or_load_index(all_chunks, embedder)

    # ── Load fine-tuned SLM ───────────────────────────────────
    model, tokenizer = load_model()

    # ── Start chat ────────────────────────────────────────────
    chat_loop(model, tokenizer, index, chunks, embedder)


if __name__ == "__main__":
    main()
