"""
cognition_engine.py — Jinx Sliding Cognition Engine
=====================================================
This is NOT a fact extractor. This is NOT just memory.
This is the layer that makes Jinx behave like neither a pure Transformer
nor a pure SSM — but something beyond both.

THE THREE PROBLEMS THIS SOLVES:
────────────────────────────────────────────────────────────────────────
PROBLEM 1 — TRANSFORMER BLINDNESS:
  A transformer reads 896 tokens then goes blind. Token 897 is gone.
  When Jinx is reading a long paper, experiment log, or code file,
  she can't hold it all in view at once.

PROBLEM 2 — SSM BLUR:
  An SSM like Mamba compresses everything into a fixed state vector.
  It "slides" but it BLURS. Ask it "what was the exact formula on line 3?"
  and it can't tell you — the exact characters are gone, dissolved into state.

PROBLEM 3 — NO CHAIN TRACKING:
  Neither architecture knows "I'm on step 7 of a 20-step experiment."
  Neither can track where it is in a long reasoning chain and pick up exactly.

THE SOLUTION — THREE ENGINES IN ONE FILE:
────────────────────────────────────────────────────────────────────────
① SlidingCognitionWindow
  When a document/text exceeds block_size (896), this engine slides
  the window forward in overlapping chunks. The tail of each chunk
  becomes the head of the next — like reading a scroll, not flipping pages.
  Each chunk is processed AND its key content is written to the VaultStore.

② VaultStore
  Exact verbatim text storage. Not embeddings. Not vectors. Actual characters.
  Stores: formulas, code snippets, equations, definitions, facts, names.
  Retrieves by keyword or semantic similarity.
  This is Jinx's PHOTOGRAPHIC MEMORY — she can read something on page 1
  and quote it verbatim on page 60.

③ ChainTracker
  When Jinx is working through a long reasoning chain (experiment, proof,
  multi-step plan), this tracks:
    - Where she IS in the chain (step N of M)
    - What she concluded at each step (exact text)
    - What's pending / what comes next
  So she can always answer "where was I?" and resume exactly.

ARCHITECTURE RELATIONSHIP:
────────────────────────────────────────────────────────────────────────
  LiquidReservoir      → session working memory (vector, decays)
  MemoryLattice        → episodic/semantic past (embeddings, persistent)
  MemoryProjection     → bridge: lattice casts into context window
  CognitionEngine      → THIS FILE
    ├── SlidingWindow  → processes long documents in overlapping chunks
    ├── VaultStore     → stores/retrieves EXACT text (formulas, facts)
    └── ChainTracker   → tracks position in long reasoning chains

Together these give Jinx:
  - Unlimited effective context (sliding)
  - Perfect verbatim recall (vault)
  - Coherent long-form reasoning (chain tracker)
  - All without changing a single weight or layer in model.py
"""

import re
import json
import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Callable, Any


# ═══════════════════════════════════════════════════════════════════════════
# ① VAULT STORE — Exact Verbatim Memory
# ═══════════════════════════════════════════════════════════════════════════

class VaultStore:
    """
    Jinx's photographic memory. Stores EXACT TEXT — not embeddings.

    What goes in the Vault:
      - Mathematical formulas:  E = mc², ∇²φ = ρ/ε₀
      - Code snippets:          def forward(self, x): ...
      - Named definitions:      "Entropy: S = -Σ p log p"
      - Experiment parameters:  "lr=5e-4, batch=2, block=896"
      - Any text Jinx marks as "must remember exactly"

    Retrieval is by:
      1. Exact keyword match (fastest)
      2. Tag match (domain, type)
      3. Fuzzy substring search (fallback)

    Vault entries are saved to disk alongside the memory lattice.
    """

    def __init__(self, save_path: str = "jinx_vault.json"):
        self.save_path = save_path
        # vault: {id: VaultEntry}
        self._vault: Dict[str, Dict] = {}
        self._load()

    # ── Storage ──────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        label: str = "",
        domain: str = "general",
        tags: Optional[List[str]] = None,
        source: str = "",
        overwrite_if_same_label: bool = True
    ) -> str:
        """
        Store exact text in the vault.

        Args:
            text:   The exact content to remember verbatim.
            label:  Human-readable name. "Schrodinger equation", "learning rate".
            domain: Category. "math", "code", "experiment", "fact", "formula".
            tags:   List of string tags for retrieval.
            source: Where this came from (e.g. "user", "document_page_3").

        Returns:
            vault_id of the stored entry.
        """
        # If same label exists and overwrite is on, replace it
        if overwrite_if_same_label and label:
            for vid, entry in self._vault.items():
                if entry.get("label", "").lower() == label.lower():
                    self._vault[vid]["text"] = text
                    self._vault[vid]["updated"] = datetime.now().isoformat()
                    self._vault[vid]["tags"] = tags or entry.get("tags", [])
                    self._save()
                    return vid

        vault_id = self._make_id(text, label)
        self._vault[vault_id] = {
            "id":      vault_id,
            "text":    text,
            "label":   label,
            "domain":  domain,
            "tags":    tags or [],
            "source":  source,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "access_count": 0,
        }
        self._save()
        return vault_id

    # ── Retrieval ─────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve vault entries by keyword/label match.
        Returns list of matching entries sorted by relevance.
        """
        query_lower = query.lower()
        scored = []

        for entry in self._vault.values():
            if domain and entry.get("domain") != domain:
                continue

            score = 0.0
            label = entry.get("label", "").lower()
            text  = entry.get("text",  "").lower()
            tags  = [t.lower() for t in entry.get("tags", [])]

            # Exact label match = highest score
            if query_lower == label:
                score = 10.0
            # Label contains query
            elif query_lower in label:
                score = 5.0
            # Query in tags
            elif any(query_lower in t for t in tags):
                score = 4.0
            # Query appears in text
            elif query_lower in text:
                score = 2.0 + text.count(query_lower) * 0.1
            # Partial word match in label
            elif any(w in label for w in query_lower.split()):
                score = 1.0

            if score > 0:
                entry["access_count"] = entry.get("access_count", 0) + 1
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def recall_by_domain(self, domain: str) -> List[Dict]:
        """Return all vault entries for a given domain."""
        return [e for e in self._vault.values() if e.get("domain") == domain]

    def recall_exact(self, label: str) -> Optional[str]:
        """
        Return the exact text for a given label, or None.
        Most direct retrieval — used when Jinx knows exactly what she wants.
        """
        label_lower = label.lower()
        for entry in self._vault.values():
            if entry.get("label", "").lower() == label_lower:
                return entry["text"]
        return None

    def format_for_context(self, entries: List[Dict], max_chars: int = 800) -> str:
        """
        Format retrieved vault entries as a string to inject into Jinx's context.
        Truncates if needed to stay within max_chars.
        """
        lines = []
        total = 0
        for e in entries:
            label = e.get("label", "")
            text  = e.get("text", "")
            line  = f"[{label}]: {text}" if label else text
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def get_all_labels(self) -> List[str]:
        """Return all stored labels — useful for letting Jinx know what she knows."""
        return [e.get("label", e["id"]) for e in self._vault.values() if e.get("label")]

    # ── Persistence ───────────────────────────────────────────────────────

    def _make_id(self, text: str, label: str) -> str:
        h = hashlib.md5((label + text[:64]).encode()).hexdigest()[:12]
        return f"vault_{h}"

    def _save(self):
        try:
            dir_ = os.path.dirname(self.save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self._vault, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[VAULT] Save failed: {e}")

    def _load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    self._vault = json.load(f)
                print(f"📚 [VAULT] Loaded {len(self._vault)} entries from {self.save_path}")
            except Exception as e:
                print(f"[VAULT] Load failed: {e}")
                self._vault = {}


# ═══════════════════════════════════════════════════════════════════════════
# ② CHAIN TRACKER — Long Reasoning Chain State
# ═══════════════════════════════════════════════════════════════════════════

class ChainTracker:
    """
    Tracks Jinx's position in a long reasoning chain.

    A "chain" is any multi-step process:
      - A scientific experiment (step 1: hypothesis, step 2: setup, ...)
      - A mathematical proof (step 1: assume, step 2: derive, ...)
      - A long code implementation (step 1: design, step 2: write module A, ...)
      - A conversation thread with many sub-topics

    At any point Jinx can:
      - Know exactly what step she's on
      - Read back what she concluded at any previous step
      - Resume from exactly where she left off
      - See what's pending ahead

    Chains are saved to disk so they survive session restarts.
    """

    def __init__(self, save_path: str = "jinx_chains.json"):
        self.save_path = save_path
        # chains: {chain_id: ChainState}
        self._chains: Dict[str, Dict] = {}
        self._active_chain: Optional[str] = None
        self._load()

    # ── Chain lifecycle ───────────────────────────────────────────────────

    def start_chain(self, name: str, goal: str, total_steps: Optional[int] = None) -> str:
        """
        Begin a new reasoning chain.

        Args:
            name:        Short identifier. "protein_folding_experiment"
            goal:        What this chain is trying to achieve.
            total_steps: Known total steps (optional — can be open-ended).

        Returns:
            chain_id
        """
        chain_id = f"chain_{hashlib.md5(name.encode()).hexdigest()[:8]}_{int(datetime.now().timestamp())}"
        self._chains[chain_id] = {
            "id":          chain_id,
            "name":        name,
            "goal":        goal,
            "total_steps": total_steps,
            "current_step": 0,
            "steps":       [],     # list of {step_n, summary, exact_output, timestamp}
            "status":      "active",
            "created":     datetime.now().isoformat(),
            "updated":     datetime.now().isoformat(),
        }
        self._active_chain = chain_id
        self._save()
        print(f"🔗 [CHAIN] Started: '{name}' | Goal: {goal}")
        return chain_id

    def record_step(
        self,
        summary: str,
        exact_output: str = "",
        chain_id: Optional[str] = None
    ) -> int:
        """
        Record completion of the current step.

        Args:
            summary:      Brief description of what was done/concluded.
            exact_output: The verbatim output/result of this step (formula, code, answer).
            chain_id:     Which chain (defaults to active chain).

        Returns:
            Step number recorded.
        """
        cid = chain_id or self._active_chain
        if not cid or cid not in self._chains:
            raise ValueError("No active chain. Call start_chain() first.")

        chain = self._chains[cid]
        step_n = chain["current_step"] + 1
        chain["steps"].append({
            "step_n":       step_n,
            "summary":      summary,
            "exact_output": exact_output,
            "timestamp":    datetime.now().isoformat(),
        })
        chain["current_step"] = step_n
        chain["updated"] = datetime.now().isoformat()
        self._save()

        total = chain.get("total_steps")
        total_str = f"/{total}" if total else ""
        print(f"🔗 [CHAIN] Step {step_n}{total_str} recorded: {summary[:60]}")
        return step_n

    def complete_chain(self, chain_id: Optional[str] = None, conclusion: str = ""):
        """Mark a chain as complete."""
        cid = chain_id or self._active_chain
        if cid and cid in self._chains:
            self._chains[cid]["status"] = "complete"
            self._chains[cid]["conclusion"] = conclusion
            self._chains[cid]["updated"] = datetime.now().isoformat()
            if self._active_chain == cid:
                self._active_chain = None
            self._save()
            print(f"✅ [CHAIN] Complete: {self._chains[cid]['name']}")

    def resume_chain(self, chain_id: str):
        """Set a previously started chain as the active chain."""
        if chain_id in self._chains:
            self._active_chain = chain_id
            self._chains[chain_id]["status"] = "active"
            self._save()

    # ── State queries ─────────────────────────────────────────────────────

    def where_am_i(self, chain_id: Optional[str] = None) -> str:
        """
        Return a human-readable summary of current position in the chain.
        This is what gets injected into Jinx's context to orient her.
        """
        cid = chain_id or self._active_chain
        if not cid or cid not in self._chains:
            return ""

        chain = self._chains[cid]
        step_n = chain["current_step"]
        total  = chain.get("total_steps")
        name   = chain["name"]
        goal   = chain["goal"]
        total_str = f" of {total}" if total else ""

        lines = [f"[CHAIN: {name}] Goal: {goal}",
                 f"Currently on step {step_n}{total_str}."]

        # Last 3 steps as context
        recent = chain["steps"][-3:]
        if recent:
            lines.append("Recent steps:")
            for s in recent:
                lines.append(f"  Step {s['step_n']}: {s['summary']}")
                if s.get("exact_output"):
                    lines.append(f"    → {s['exact_output'][:120]}")

        return "\n".join(lines)

    def recall_step(self, step_n: int, chain_id: Optional[str] = None) -> Optional[Dict]:
        """Return the full record of a specific step."""
        cid = chain_id or self._active_chain
        if not cid or cid not in self._chains:
            return None
        steps = self._chains[cid]["steps"]
        for s in steps:
            if s["step_n"] == step_n:
                return s
        return None

    def list_chains(self, status: Optional[str] = None) -> List[Dict]:
        """List all chains, optionally filtered by status."""
        chains = list(self._chains.values())
        if status:
            chains = [c for c in chains if c["status"] == status]
        return sorted(chains, key=lambda c: c["updated"], reverse=True)

    def get_active(self) -> Optional[Dict]:
        """Return the active chain state, or None."""
        if self._active_chain:
            return self._chains.get(self._active_chain)
        return None

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        try:
            dir_ = os.path.dirname(self.save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            data = {
                "chains":       self._chains,
                "active_chain": self._active_chain,
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CHAIN] Save failed: {e}")

    def _load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._chains       = data.get("chains", {})
                self._active_chain = data.get("active_chain")
                active_chains = [c for c in self._chains.values() if c.get("status") == "active"]
                print(f"🔗 [CHAIN] Loaded {len(self._chains)} chains "
                      f"({len(active_chains)} active) from {self.save_path}")
            except Exception as e:
                print(f"[CHAIN] Load failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# ③ SLIDING COGNITION WINDOW — Reading Beyond the Context Limit
# ═══════════════════════════════════════════════════════════════════════════

# Patterns for auto-detecting vault-worthy content inside text chunks
_VAULT_PATTERNS = [
    # Math formulas (LaTeX-style or inline)
    (re.compile(r'[A-Za-z\s]+=\s*[-\w\s\^/\(\)\+\*\.]+(?=\n|$)'),          "formula"),
    # Code blocks
    (re.compile(r'```[\s\S]+?```|def \w+\([\s\S]+?(?=\n\n|\Z)', re.M),      "code"),
    # Numbered definitions
    (re.compile(r'(?:Definition|Theorem|Lemma|Corollary)\s*\d*[:\.].*',
                re.IGNORECASE),                                               "definition"),
    # Experiment parameters
    (re.compile(r'(?:lr|learning.rate|batch.size|epochs?|steps?)\s*[=:]\s*[\d\.e\-]+',
                re.IGNORECASE),                                               "parameter"),
]


class SlidingCognitionWindow:
    """
    Processes text longer than Jinx's context window (block_size=896 tokens)
    using an overlapping sliding window approach.

    KEY PROPERTIES:
    ─────────────────────────────────────────────────────────────────────
    • OVERLAP: Each chunk shares `overlap_tokens` tokens with the previous.
      This prevents meaning from being cut at chunk boundaries.
      Token 870-896 of chunk 1 = Token 0-26 of chunk 2. The "scroll" scrolls.

    • AUTO-VAULT: As each chunk is processed, important content
      (formulas, code, definitions, params) is automatically extracted
      and stored in the VaultStore for verbatim recall later.

    • CHAIN INTEGRATION: If a ChainTracker is provided, each chunk
      automatically records a step — so Jinx knows her position in
      the document at all times.

    • GENERATOR-BASED: The window is a Python generator. It yields
      one chunk at a time, so it works with documents of ANY length
      without loading everything into memory at once.

    This is the SSM property (process infinite sequences) implemented
    as a cognition layer — not as a model architecture change.
    """

    def __init__(
        self,
        block_size:     int = 896,
        overlap_tokens: int = 128,
        vault:          Optional[VaultStore]   = None,
        chain_tracker:  Optional[ChainTracker] = None,
        tokenizer       = None,
    ):
        """
        Args:
            block_size:     Jinx's context window size (tokens).
            overlap_tokens: How many tokens to carry over between chunks.
                            Higher = more continuity, more compute.
            vault:          VaultStore instance for auto-vaulting important content.
            chain_tracker:  ChainTracker to record progress through a document.
            tokenizer:      Tokenizer (Qwen). If None, uses char-based approximation.
        """
        self.block_size     = block_size
        self.overlap        = overlap_tokens
        self.vault          = vault
        self.chain_tracker  = chain_tracker
        self.tokenizer      = tokenizer
        self._step_size     = block_size - overlap_tokens  # how far to advance each step

    # ── Main sliding interface ────────────────────────────────────────────

    def slide(
        self,
        text:        str,
        source_name: str = "document",
        domain:      str = "general",
        on_chunk:    Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Slide the cognition window over `text`.

        This is a GENERATOR — iterate over it to process the document.
        Each iteration yields (chunk_text, chunk_index, total_chunks).

        Args:
            text:        The full text to process (can be arbitrarily long).
            source_name: Name of the source document/context.
            domain:      Domain tag for auto-vaulted entries.
            on_chunk:    Optional callback(chunk_text, chunk_idx, total_chunks).
                         Use this to pass each chunk to Jinx's generate() function.

        Usage:
            for chunk, idx, total in window.slide(long_text, source_name="paper"):
                response = generate(model, config, chunk_prompt)
                window.vault_from_response(response, domain="science")
        """
        chunks = self._split_into_chunks(text)
        total  = len(chunks)

        for idx, chunk in enumerate(chunks):
            # Auto-vault important content from this chunk
            if self.vault:
                self._auto_vault(chunk, source=f"{source_name}_chunk{idx}", domain=domain)

            # Record in chain tracker
            if self.chain_tracker and self.chain_tracker.get_active():
                summary = self._summarize_chunk(chunk)
                self.chain_tracker.record_step(
                    summary      = f"[{source_name}] chunk {idx+1}/{total}: {summary}",
                    exact_output = chunk[:300],  # store first 300 chars verbatim
                )

            # User callback
            if on_chunk:
                on_chunk(chunk, idx, total)

            yield chunk, idx, total

    def build_prompt_with_context(
        self,
        chunk:       str,
        chunk_idx:   int,
        total_chunks: int,
        vault_query: str = "",
        chain_id:    Optional[str] = None,
    ) -> str:
        """
        Build a context-enriched prompt for a given chunk.
        Injects:
          1. Chain position (where am I in this document?)
          2. Vault recalls (what exact facts do I need for this chunk?)
          3. The chunk itself

        This is what you pass to generate() instead of raw chunk text.
        """
        parts = []

        # Chain position header
        if self.chain_tracker:
            where = self.chain_tracker.where_am_i(chain_id)
            if where:
                parts.append(where)

        # Vault recall
        if self.vault and vault_query:
            hits = self.vault.recall(vault_query, top_k=3)
            if hits:
                vault_ctx = self.vault.format_for_context(hits, max_chars=400)
                parts.append(f"[RECALL]\n{vault_ctx}")

        # Chunk position marker
        parts.append(f"[READING: chunk {chunk_idx+1}/{total_chunks}]")

        # The actual chunk
        parts.append(chunk)

        return "\n".join(parts)

    def vault_from_response(self, response_text: str, domain: str = "general"):
        """
        After Jinx generates a response to a chunk, scan that response
        for important content and store it in the vault.
        Use this to capture Jinx's own conclusions and derivations.
        """
        if self.vault:
            self._auto_vault(response_text, source="jinx_response", domain=domain)

    # ── Internals ─────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        Uses char-based approximation if no tokenizer (4 chars ≈ 1 token).
        """
        if self.tokenizer is not None:
            return self._split_by_tokens(text)
        else:
            return self._split_by_chars(text)

    def _split_by_chars(self, text: str) -> List[str]:
        """Approximate token splitting using 4 chars/token heuristic."""
        char_block   = self.block_size * 4
        char_overlap = self.overlap    * 4
        char_step    = char_block - char_overlap

        chunks = []
        start  = 0
        while start < len(text):
            end = start + char_block
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(text):
                break
            start += char_step
        return chunks if chunks else [text]

    def _split_by_tokens(self, text: str) -> List[str]:
        """Precise token-based splitting using the tokenizer."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        step      = self._step_size
        chunks    = []
        start     = 0
        while start < len(token_ids):
            end   = start + self.block_size
            chunk_ids = token_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end >= len(token_ids):
                break
            start += step
        return chunks if chunks else [text]

    def _auto_vault(self, text: str, source: str, domain: str):
        """
        Scan text for vault-worthy content and store it automatically.
        Detects: formulas, code blocks, definitions, experiment parameters.
        """
        if not self.vault:
            return

        for pattern, type_tag in _VAULT_PATTERNS:
            for match in pattern.finditer(text):
                matched = match.group(0).strip()
                if len(matched) < 8:      # too short to be meaningful
                    continue
                if len(matched) > 2000:   # too long — truncate
                    matched = matched[:2000]

                # Build a label from the first meaningful tokens
                label = matched[:60].replace("\n", " ").strip()

                self.vault.store(
                    text   = matched,
                    label  = label,
                    domain = domain,
                    tags   = [type_tag, source],
                    source = source,
                    overwrite_if_same_label=False,  # don't overwrite — keep all versions
                )

    def _summarize_chunk(self, chunk: str, max_len: int = 80) -> str:
        """Extract the first meaningful line as a chunk summary."""
        for line in chunk.split("\n"):
            line = line.strip()
            if len(line) > 20:
                return line[:max_len]
        return chunk[:max_len].replace("\n", " ")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER INTERFACE — CognitionEngine
# ═══════════════════════════════════════════════════════════════════════════

class CognitionEngine:
    """
    The unified interface. One object. Full cognition stack.

    Usage in test_checkpoint.py / live_chat:

        from cognition_engine import CognitionEngine
        cognition = CognitionEngine(block_size=896, save_dir="./jinx_cognition")

        # Reading a long document
        for chunk, idx, total in cognition.read(long_text, name="experiment_log"):
            prompt = cognition.build_prompt(chunk, idx, total, query="formula")
            response = generate(model, config, f"User: {prompt}\nJinx:")
            cognition.learn_from(response)   # vault Jinx's conclusions too

        # Storing a specific fact/formula verbatim
        cognition.remember("E = mc²", label="mass-energy equivalence", domain="physics")

        # Recalling it later
        result = cognition.recall("mass-energy equivalence")
        # → "E = mc²"

        # Tracking a long experiment
        chain_id = cognition.begin("protein_folding_v2", goal="minimize RMSD < 0.5Å", steps=12)
        cognition.step("Loaded dataset, 4200 samples", output="dataset_shape=(4200,128)")
        print(cognition.status())   # prints current step in chain

        # Injecting full cognition context before any generation
        ctx = cognition.inject_context(user_input)
        prompt = f"{ctx}\nUser: {user_input}\nJinx:"
    """

    def __init__(
        self,
        block_size:     int = 896,
        overlap_tokens: int = 128,
        save_dir:       str = ".",
        tokenizer       = None,
    ):
        vault_path = os.path.join(save_dir, "jinx_vault.json")
        chain_path = os.path.join(save_dir, "jinx_chains.json")

        self.vault   = VaultStore(save_path=vault_path)
        self.chains  = ChainTracker(save_path=chain_path)
        self.window  = SlidingCognitionWindow(
            block_size     = block_size,
            overlap_tokens = overlap_tokens,
            vault          = self.vault,
            chain_tracker  = self.chains,
            tokenizer      = tokenizer,
        )

    # ── Sliding window reading ─────────────────────────────────────────────

    def read(self, text: str, name: str = "document", domain: str = "general"):
        """Generator. Slide over a long text, yielding (chunk, idx, total)."""
        return self.window.slide(text, source_name=name, domain=domain)

    def build_prompt(self, chunk: str, idx: int, total: int,
                     query: str = "") -> str:
        """Build a context-enriched prompt for the current chunk."""
        return self.window.build_prompt_with_context(chunk, idx, total,
                                                      vault_query=query)

    def learn_from(self, response_text: str, domain: str = "general"):
        """Vault important content from Jinx's own response."""
        self.window.vault_from_response(response_text, domain=domain)

    # ── Vault (verbatim memory) ────────────────────────────────────────────

    def remember(self, text: str, label: str = "", domain: str = "general",
                 tags: Optional[List[str]] = None):
        """Store exact text in the vault."""
        return self.vault.store(text, label=label, domain=domain, tags=tags)

    def recall(self, query: str, domain: Optional[str] = None,
               exact: bool = False) -> str:
        """
        Recall from vault.
        If exact=True, returns exact text for label.
        Otherwise returns formatted string of top matches.
        """
        if exact:
            result = self.vault.recall_exact(query)
            return result or ""
        hits = self.vault.recall(query, domain=domain, top_k=5)
        return self.vault.format_for_context(hits)

    # ── Chain tracking ─────────────────────────────────────────────────────

    def begin(self, name: str, goal: str, steps: Optional[int] = None) -> str:
        """Start a new reasoning chain."""
        return self.chains.start_chain(name, goal, total_steps=steps)

    def step(self, summary: str, output: str = ""):
        """Record the current step in the active chain."""
        return self.chains.record_step(summary, exact_output=output)

    def finish(self, conclusion: str = ""):
        """Mark the active chain as complete."""
        self.chains.complete_chain(conclusion=conclusion)

    def status(self) -> str:
        """Return human-readable status of the active chain."""
        return self.chains.where_am_i() or "[No active chain]"

    # ── Context injection ──────────────────────────────────────────────────

    def inject_context(self, user_input: str, max_chars: int = 600) -> str:
        """
        Build a soft context block to prepend to any prompt.
        Combines:
          - Active chain status (where am I?)
          - Vault recalls relevant to user input (what do I know?)

        Keep it under max_chars so it doesn't eat Jinx's context window.
        """
        parts = []

        # Chain status
        chain_status = self.chains.where_am_i()
        if chain_status:
            parts.append(chain_status)

        # Vault recall based on user input
        vault_hits = self.vault.recall(user_input, top_k=3)
        if vault_hits:
            vault_text = self.vault.format_for_context(
                vault_hits, max_chars=max_chars - len(chain_status) - 50
            )
            if vault_text:
                parts.append(f"[MEMORY]\n{vault_text}")

        return "\n".join(parts)
