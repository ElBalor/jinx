"""
fact_extractor.py — Jinx Personal Fact Memory
==============================================
Detects personal facts in conversation turns (name, age, location, etc.)
and stores them as READABLE TEXT in the MemoryLattice semantic layer.

This gives Jinx two kinds of memory working together:

    Semantic facts  (this file)  →  "user.name = Eric"   ← readable text
    Episodic memory (lattice)    →  compressed embeddings ← geometric feel
    LiquidReservoir              →  session momentum      ← working memory

Together these form Jinx's full memory stack:

    ┌─────────────────────────────────────────────────────┐
    │  What she knows (facts)  ←  fact_extractor.py       │
    │  What she felt (episodes)←  memory_lattice.py       │
    │  What she's thinking now ←  LiquidReservoir         │
    │  What she sees beyond    ←  MemoryProjection        │
    │  Who she IS              ←  slow weights (DNA)       │
    └─────────────────────────────────────────────────────┘

Usage (in chat_jinx.py or test_checkpoint.py):
    from fact_extractor import FactExtractor
    extractor = FactExtractor(model.memory_lattice)

    # After each user turn:
    facts = extractor.extract_and_store(user_text, context_embedding)

    # To recall facts before generating:
    recall = extractor.recall_all()
    # → "User's name is Eric. User is 21 years old. User lives in Abuja."
"""

import re
import torch
from datetime import datetime
from typing import Optional, Dict, List


# ── Fact patterns ────────────────────────────────────────────────────────────
# Each entry: (slot_key, compiled_regex, value_group_index)
# slot_key  = how it's stored: "user.name", "user.age", etc.
# regex     = pattern to detect the fact in natural language
# group     = which capture group holds the value

FACT_PATTERNS: List[tuple] = [

    # Name
    ("user.name",
     re.compile(
         r"(?:my name is|i(?:'m| am) called|call me|people call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
         re.IGNORECASE
     ), 1),

    # Age
    ("user.age",
     re.compile(
         r"(?:i(?:'m| am)\s+(\d{1,3})\s*(?:years?\s*old)?|"
         r"my age is\s+(\d{1,3})|"
         r"i(?:'m| am)\s+(\d{1,3}))",
         re.IGNORECASE
     ), None),   # None = scan all groups

    # Location / city
    ("user.location",
     re.compile(
         r"(?:i(?:'m| am) from|i live in|i(?:'m| am) based in|i(?:'m| am) in)\s+"
         r"([A-Z][a-zA-Z\s,]+?)(?:\.|,|\s*$)",
         re.IGNORECASE
     ), 1),

    # Occupation / job
    ("user.job",
     re.compile(
         r"(?:i(?:'m| am) a(?:n)?\s+|i work as a(?:n)?\s+|my job is\s+|i(?:'m| am) working as\s+)"
         r"([a-zA-Z\s]+?)(?:\.|,|\s*$)",
         re.IGNORECASE
     ), 1),

    # Creator / project relationship
    ("user.role",
     re.compile(
         r"(?:i(?:'m| am) your creator|i built you|i made you|i created you|"
         r"i(?:'m| am) eric yaka|i(?:'m| am) the inventor)",
         re.IGNORECASE
     ), 0),   # 0 = use the full match as value

    # Preferred name for Jinx to use
    ("user.preferred_address",
     re.compile(
         r"(?:call me|address me as|refer to me as)\s+([A-Za-z]+)",
         re.IGNORECASE
     ), 1),

    # Hobby / interest
    ("user.interest",
     re.compile(
         r"(?:i love|i enjoy|i(?:'m| am) into|i like|i(?:'m| am) passionate about)\s+"
         r"([a-zA-Z\s]+?)(?:\.|,|\s*$)",
         re.IGNORECASE
     ), 1),

    # Language
    ("user.language",
     re.compile(
         r"(?:i speak|my language is|my native language is)\s+([A-Za-z]+)",
         re.IGNORECASE
     ), 1),
]


class FactExtractor:
    """
    Scans user utterances for personal facts and stores them
    in the MemoryLattice semantic layer as readable text entries.

    Facts are stored with:
        domain   = "user_facts"
        content  = "user.name = Eric"        ← human-readable
        metadata = {"slot": "user.name",
                    "value": "Eric",
                    "raw": "my name is Eric"} ← original utterance
        tags     = {"type": "fact",
                    "confidence": 1.0}
    """

    def __init__(self, memory_lattice, device: str = "cpu"):
        self.lattice = memory_lattice
        self.device  = device
        # In-memory cache of known facts (slot → value) for fast recall
        self._fact_cache: Dict[str, str] = {}
        # Load any facts already in the lattice
        self._rebuild_cache()

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_and_store(
        self,
        user_text: str,
        context_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, str]:
        """
        Scan `user_text` for personal facts.
        Store any new/updated facts in the lattice.
        Returns dict of newly extracted facts {slot: value}.
        """
        found = {}

        for slot, pattern, group in FACT_PATTERNS:
            match = pattern.search(user_text)
            if not match:
                continue

            # Extract value
            if group is None:
                # Scan all groups for the first non-None
                value = next((g for g in match.groups() if g), None)
            elif group == 0:
                # Special: role detection — value is canonical label
                value = self._canonical_role(match.group(0))
            else:
                value = match.group(group) if match.lastindex and match.lastindex >= group else None

            if not value:
                continue
            value = value.strip().rstrip(".,")

            # Don't re-store identical fact
            if self._fact_cache.get(slot) == value:
                continue

            # Store in lattice
            content = f"{slot} = {value}"
            self.lattice.store_semantic(
                content   = content,
                domain    = "user_facts",
                confidence= 1.0,
                context_embedding = context_embedding,
                tags      = {"type": "fact", "slot": slot,
                             "value": value, "confidence": 1.0}
            )

            # Update cache
            self._fact_cache[slot] = value
            found[slot] = value

        if found:
            _slots = ", ".join(f"{k}={v}" for k, v in found.items())
            print(f"🧠 [FACTS] Stored: {_slots}")

        return found

    def recall_all(self) -> str:
        """
        Return a natural language summary of all known facts.
        Suitable for prepending to a prompt as soft context.

        Example output:
            "User's name is Eric. User is 21 years old. User lives in Abuja."
        """
        if not self._fact_cache:
            return ""
        parts = []
        for slot, value in self._fact_cache.items():
            parts.append(self._verbalize(slot, value))
        return " ".join(p for p in parts if p)

    def recall_slot(self, slot: str) -> Optional[str]:
        """Return the stored value for a specific slot, or None."""
        return self._fact_cache.get(slot)

    def get_facts_dict(self) -> Dict[str, str]:
        """Return a copy of the full fact cache."""
        return dict(self._fact_cache)

    # ── Internals ────────────────────────────────────────────────────────────

    def _rebuild_cache(self):
        """Reload facts already stored in the lattice into the in-memory cache."""
        domain_ids = self.lattice.domain_index.get("user_facts", [])
        for mid in domain_ids:
            entry = self.lattice.semantic_memories.get(mid)
            if entry and entry.tags and entry.tags.get("type") == "fact":
                slot  = entry.tags.get("slot")
                value = entry.tags.get("value")
                if slot and value:
                    # Keep the most recent value if multiple entries for same slot
                    if slot not in self._fact_cache:
                        self._fact_cache[slot] = value
                    else:
                        # Compare timestamps — keep newer
                        existing_ts = ""
                        for mid2 in domain_ids:
                            e2 = self.lattice.semantic_memories.get(mid2)
                            if e2 and e2.tags and e2.tags.get("slot") == slot:
                                if e2.timestamp > existing_ts:
                                    existing_ts = e2.timestamp
                                    self._fact_cache[slot] = e2.tags.get("value", value)

    def _canonical_role(self, match_text: str) -> str:
        """Map matched text to a canonical role label."""
        t = match_text.lower()
        if "creator" in t or "built" in t or "made" in t or "created" in t:
            return "creator"
        if "eric yaka" in t:
            return "Eric Yaka (creator)"
        if "inventor" in t:
            return "inventor"
        return "creator"

    def _verbalize(self, slot: str, value: str) -> str:
        """Convert a slot/value pair into a natural language sentence."""
        templates = {
            "user.name":              f"User's name is {value}.",
            "user.age":               f"User is {value} years old.",
            "user.location":          f"User is based in {value}.",
            "user.job":               f"User works as a {value}.",
            "user.role":              f"User is Jinx's {value}.",
            "user.preferred_address": f"User prefers to be called {value}.",
            "user.interest":          f"User is interested in {value}.",
            "user.language":          f"User speaks {value}.",
        }
        return templates.get(slot, f"{slot.replace('.', ' ')} is {value}.")
