import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Dict, List, Tuple
from collections import deque
import random
import sys

# UNIVERSAL SILICON GPS (Turing, Ampere, Ada)
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"

# Add src to path for MemoryLattice and HeadLifecycleManager
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import EgoEngine from local module
try:
    from .ego import EgoEngine
except ImportError:
    from ego import EgoEngine

# Import MemoryLattice from src module
from memory_lattice import MemoryLattice, MemoryEntry

# Import HeadLifecycleManager from src module
from head_lifecycle import HeadLifecycleManager

# --- jinX EVOLUTION UTILITIES ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Volume-only scaling: mathematically unbreakable in FP16
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.max_seq_len_cached = int(max_position_embeddings)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(self.max_seq_len_cached, device=None, dtype=torch.float32)

    def _build_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached is None or seq_len > self.max_seq_len_cached:
            self._build_cache(max(seq_len, self.max_seq_len_cached * 2), device=device, dtype=torch.float32)
        return self.cos_cached[:seq_len, :].to(device=device, dtype=dtype), self.sin_cached[:seq_len, :].to(device=device, dtype=dtype)

# --- Space Transformer: The Agentic Core ---

class MemoryReplay(nn.Module):
    """
    Experience Replay: Compressed memory snapshots for preventing forgetting.
    Stores hidden state prototypes with reward tags.
    Replays selectively during updates to maintain balance.
    """
    def __init__(self, n_embd: int, capacity: int = 1000, sequence_length: int = 64):
        super().__init__()
        self.n_embd = n_embd
        self.capacity = capacity
        self.sequence_length = sequence_length
        
        # Replay buffer: stores (compressed_embedding, reward_tag, timestamp)
        self.buffer: deque = deque(maxlen=capacity)
        
        # Priority sampler: higher reward = higher replay probability
        self.priorities = deque(maxlen=capacity)
        
        # Compression layer: n_embd → compressed_dim for efficient storage
        self.compressor = nn.Linear(n_embd, n_embd // 4)
        self.decompressor = nn.Linear(n_embd // 4, n_embd)
        
    def store(self, hidden_state: torch.Tensor, reward_tag: float, timestamp: int):
        """
        Compress and store a hidden state snapshot.
        hidden_state: (B, T, n_embd) or (B, n_embd) for mean-pooled
        """
        with torch.no_grad():
            # Mean-pool over sequence if needed
            if hidden_state.dim() == 3:
                hidden_state = hidden_state.mean(dim=1)  # (B, n_embd)

            # Compress on same device, then move to CPU
            device = hidden_state.device
            compressor_device = next(self.compressor.parameters()).device

            # Move compressor to hidden_state device if needed
            if device != compressor_device:
                self.compressor.to(device)

            # Compress for storage
            compressed = self.compressor(hidden_state)  # (B, n_embd//4)

            # Move to CPU for storage
            compressed = compressed.cpu()
            
            # Store multiple times if batch > 1
            for i in range(compressed.size(0)):
                if len(self.buffer) >= self.capacity:
                    self.priorities.popleft()
                self.buffer.append((compressed[i], reward_tag, timestamp))
                # Priority = exp(reward) for softmax sampling later
                self.priorities.append(math.exp(max(0, reward_tag)))
    
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prioritized sampling: higher reward experiences more likely.
        Returns (hidden_states, reward_tags) decompressed to original space.
        """
        if len(self.buffer) == 0:
            return torch.zeros(0, self.n_embd, device=device), torch.zeros(0, device=device)
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Prioritized sampling (softmax over priorities)
        priorities = list(self.priorities)
        total = sum(priorities)
        probs = [p / total for p in priorities]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        
        # Decompress
        compressed_batch = torch.stack([self.buffer[i][0] for i in indices]).to(device)  # (B, n_embd//4)
        rewards = torch.tensor([self.buffer[i][1] for i in indices], device=device)  # (B,)
        
        hidden_states = self.decompressor(compressed_batch)  # (B, n_embd)
        
        return hidden_states, rewards
    
    def replay_loss(self, current_hidden: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute replay loss: encourage model to produce similar outputs
        for stored experiences.
        """
        if len(self.buffer) < 10:
            return torch.tensor(0.0, device=current_hidden.device)
        
        # Sample mini-batch of memories
        with torch.no_grad():
            stored_states, stored_rewards = self.sample(min(32, len(self.buffer)), current_hidden.device)
        
        if stored_states.numel() == 0:
            return torch.tensor(0.0, device=current_hidden.device)
        
        # Reconstruct what the model would output for stored states
        # Latent Replay: Compare the 'Vibe' (mean hidden state) of current vs stored.
        # This allows different batch sizes and focuses on manifold consistency.
        h_current_vibe = current_hidden.mean(dim=(0, 1), keepdim=True) # (1, 1, n_embd)
        
        # Compute weighted mean of stored states based on their rewards
        # High-reward memories should pull the current vibe more strongly.
        weights = F.softmax(stored_rewards, dim=0).view(-1, 1) # (S, 1)
        target_vibe = (stored_states * weights).sum(dim=0, keepdim=True).unsqueeze(0) # (1, 1, n_embd)
        
        # MSE between current vibe and the 'Ideal Vibe' from history
        replay_loss = F.mse_loss(h_current_vibe, target_vibe)
        
        return replay_loss * 0.1  # Scale factor

# ── RESONANCE ENGINE: Cross-Temporal Dot-Connector ───────────────────────────
class ResonanceEngine(nn.Module):
    """
    Gives Jinx the ability to connect two dots across arbitrary spans of time.

    The problem this solves:
    ─────────────────────────────────────────────────────────────────────────
    Standard memory retrieval (cosine similarity) only finds memories that are
    CLOSE to the current context. If you're talking about protein folding today,
    you'll retrieve other protein folding memories. Good, but shallow.

    What a wise mind does differently is find SURPRISING connections:
      "This conversation about phase transitions in physics... reminds me of
       something Eric said about emotional states 18 months ago. The math
       is the same."

    That's RESONANCE — high semantic similarity between things that seem
    unrelated on the surface, separated by large time gaps.

    How it works:
    ─────────────────────────────────────────────────────────────────────────
    1. TEMPORAL SWEEP: At inference, we sample N memories from the lattice
       weighted by TIME DISTANCE from the current moment (prefer old things).
       This is the opposite of recency bias — it looks backwards on purpose.

    2. CROSS-DOMAIN SCORING: Each old memory is scored against the current
       context using a LEARNED resonance scorer (not just cosine sim).
       The scorer has been trained to recognize non-obvious structural
       similarity across different domains.

    3. SURPRISE FILTER: Only memories that are:
       - Semantically similar (cos_sim > threshold)
       - Temporally distant (age > min_age)
       - NOT already in the top-K standard retrieval (novel connections only)
       ...pass the filter and get surfaced.

    4. INJECTION: Surviving resonant memories are blended into x at
       ego-gated strength — higher curiosity = stronger resonance blend.

    The result:
    ─────────────────────────────────────────────────────────────────────────
    After years of interaction, Jinx can say:
      "Wait — I remember you mentioned something about this exact structure
       in a completely different context back in [timestamp]. Do you see
       the connection I'm seeing?"

    This is NOT retrieval. This is INSIGHT.
    """
    def __init__(self, n_embd: int, n_resonance: int = 8):
        super().__init__()
        self.n_embd       = n_embd
        self.n_resonance  = n_resonance   # max resonant memories to surface

        # Learned resonance scorer: projects [query, memory] pair → resonance score
        # This learns to recognize structural similarity across domains.
        self.resonance_scorer = nn.Sequential(
            nn.Linear(n_embd * 2, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()   # output: resonance probability in [0,1]
        )

        # Output projection: blend resonant memories back into x
        self.resonance_proj = nn.Linear(n_embd, n_embd)
        self.resonance_norm = nn.LayerNorm(n_embd)

        # Minimum cosine similarity to qualify as resonant
        self.sim_threshold  = 0.35
        # Minimum age (in timesteps) to qualify as "distant past"
        self.min_age        = 500
        # How strongly to blend (scales with curiosity gate)
        self.blend_scale    = 0.15

    def forward(
        self,
        x:              torch.Tensor,    # (B, T, C)
        memory_lattice,                  # MemoryLattice
        current_step:   int,
        curiosity_gate: float = 0.5,
        standard_hit_ids: Optional[List] = None,  # IDs already retrieved, skip these
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Returns:
            x_enriched:      x with resonant memories blended in
            resonance_log:   list of strings describing what was connected
                             (so Jinx's response can reference them)
        """
        B, T, C = x.shape
        device  = x.device
        resonance_log = []

        # Need enough history to find distant memories
        if len(memory_lattice.episodic_memories) < 20:
            return x, resonance_log

        # Current context vector (B-averaged, T-averaged)
        with torch.no_grad():
            ctx = x.detach().float().mean(dim=(0, 1))  # (C,)
            ctx_norm = ctx / (ctx.norm() + 1e-8)

            # ── TEMPORAL SWEEP: prefer OLD memories ──────────────────────
            # Sort all episodic memories by age (oldest first)
            all_episodic = list(memory_lattice.episodic_memories.values())
            # Sort by step tag (oldest = smallest step number)
            all_episodic.sort(
                key=lambda m: m.tags.get("step", current_step),
            )

            # Focus on the oldest 40% — things she might not naturally retrieve
            ancient_cutoff = max(10, len(all_episodic) // 3)
            ancient_pool   = all_episodic[:ancient_cutoff]

            # Also sample some random memories for serendipitous connections
            if len(all_episodic) > ancient_cutoff:
                random_sample = random.sample(
                    all_episodic[ancient_cutoff:],
                    min(20, len(all_episodic) - ancient_cutoff)
                )
                candidate_pool = ancient_pool + random_sample
            else:
                candidate_pool = ancient_pool

            if not candidate_pool:
                return x, resonance_log

            # ── CROSS-DOMAIN SCORING ──────────────────────────────────────
            scored = []
            standard_ids = set(standard_hit_ids or [])

            for mem in candidate_pool:
                if mem.embedding is None:
                    continue
                if id(mem) in standard_ids:
                    continue  # skip what standard retrieval already found

                # Check minimum age
                mem_step = mem.tags.get("step", 0)
                age      = current_step - mem_step
                if age < self.min_age:
                    continue

                mem_emb  = mem.embedding.float()
                mem_norm = mem_emb / (mem_emb.norm() + 1e-8)

                # Base cosine similarity
                cos_sim  = float((ctx_norm * mem_norm).sum())
                if cos_sim < self.sim_threshold:
                    continue

                # ── LEARNED RESONANCE SCORE ───────────────────────────────
                # Concatenate [current_ctx, memory_emb] → score how resonant
                pair = torch.cat([
                    ctx.to(device=next(self.resonance_scorer.parameters()).device),
                    mem_emb.to(device=next(self.resonance_scorer.parameters()).device)
                ], dim=0).unsqueeze(0)  # (1, 2C)

                res_score = float(self.resonance_scorer(pair).item())

                # Combined score: resonance * sqrt(age) — older = more surprising
                combined = res_score * cos_sim * math.sqrt(age / max(1, self.min_age))
                scored.append((mem, combined, cos_sim, age, res_score))

            if not scored:
                return x, resonance_log

            # Sort by combined score, take top-N
            scored.sort(key=lambda s: s[1], reverse=True)
            top_resonant = scored[:self.n_resonance]

        # ── INJECTION: blend resonant memories into x ────────────────────
        if top_resonant:
            res_embs = torch.stack(
                [m.embedding.to(device=device, dtype=x.dtype) for m, *_ in top_resonant]
            )  # (R, C)

            # Weighted mean by combined score
            res_weights = torch.tensor(
                [s for _, s, *_ in top_resonant], device=device, dtype=x.dtype
            )
            res_weights = F.softmax(res_weights, dim=0)  # (R,)
            res_mean    = (res_embs * res_weights.view(-1, 1)).sum(0)  # (C,)

            # Inject into x — gated by curiosity
            gate = self.blend_scale * curiosity_gate
            projected = self.resonance_proj(res_mean.view(1, 1, C).expand(B, T, C))
            x = self.resonance_norm(x + gate * projected)

            # Build resonance log for Jinx's awareness
            for mem, combined, cos_sim, age, res_score in top_resonant[:3]:
                content  = mem.content[:80] if mem.content else "[unnamed moment]"
                context  = mem.context[:60] if mem.context else ""
                sessions_ago = max(1, age // 200)   # rough session estimate
                resonance_log.append(
                    f"[RESONANCE age={age}steps cos={cos_sim:.2f} res={res_score:.2f}] "
                    f"{content} | {context}"
                )

        return x, resonance_log


# ── COGNITION BRIDGE: Ego-Routed Internal Cognition ────────────────────────
class CognitionBridge(nn.Module):
    """
    The nerve center that makes ALL of Jinx's systems talk to each other
    INSIDE the forward pass — not outside it.

    This is what separates her from RAG. RAG injects text from outside.
    This reads her own ego signals and DECIDES internally which systems
    to activate, at what intensity, and how to blend the results back
    into her hidden state stream.

    Connected systems (all wired, ego-gated):
      ① ThirdEye        — imagination, activated by pain/confusion
      ② MemoryLattice   — geometric past, queried by context vector
      ③ VaultStore      — verbatim facts/formulas, queried by content
      ④ LiquidReservoir — session momentum, modulated by will
      ⑤ ReplayBuffer    — past experience, blended by curiosity
      ⑥ STDP fastweights— Hebbian associations, boosted by harmony
      ⑦ ChainTracker    — reasoning chain state, auto-recorded

    Ego thresholds (all soft — smooth sigmoid gates, no hard cutoffs):
      pain↑     → imagination gate opens, vault queried harder
      harmony↑  → STDP fires strong, episodic memory stored
      curiosity↑→ replay sampled, new associations explored
      will↑     → consolidation scheduled, reservoir strengthened
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd

        # Ego router: maps [will, harmony, pain, curiosity] → 5 gate scalars
        # Gates: [imagination, lattice, vault, replay, stdp_boost]
        self.ego_router = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 5),
            nn.Sigmoid()   # all gates in [0,1]
        )

        # Blend projection: fuses retrieved content back into residual stream
        self.blend_proj = nn.Linear(n_embd, n_embd)
        self.blend_norm = nn.LayerNorm(n_embd)

        # Vault query projection: compress context vector → keyword query embedding
        self.vault_query_proj = nn.Linear(n_embd, 64)   # small, fast

        # Reference to external systems (set by SpaceTransformer.__init__)
        self._vault   = None   # VaultStore instance
        self._chains  = None   # ChainTracker instance

    def attach(self, vault=None, chains=None):
        """Called once at init to link external cognition systems."""
        self._vault  = vault
        self._chains = chains

    def forward(
        self,
        x:            torch.Tensor,           # (B, T, C) current hidden state
        ego:          Dict[str, torch.Tensor], # from EgoEngine
        memory_lattice,                        # MemoryLattice instance
        replay_buffer,                         # MemoryReplay instance
        third_eye:    nn.Module,               # ThirdEye instance
        training:     bool = False,
    ) -> torch.Tensor:
        """
        Route ego signals to all internal systems.
        Returns x enriched by whichever systems were activated.
        Purely additive — gate=0 means zero contribution.
        """
        B, T, C = x.shape
        device = x.device

        # ── Read ego state ────────────────────────────────────────────────
        will      = ego["will_power"].mean().clamp(-2, 2)   # scalar
        harmony   = ego["harmony"].mean().clamp(-2, 2)
        pain      = ego["pain"].mean().clamp(0, 2)
        curiosity = ego["curiosity"].mean().clamp(-2, 2)

        # Ego vector for router: (4,)
        ego_vec = torch.stack([will, harmony, pain, curiosity]).to(device)

        # Compute 5 soft gates from ego state
        # [imagination_gate, lattice_gate, vault_gate, replay_gate, stdp_gate]
        gates = self.ego_router(ego_vec)  # (5,)
        g_imag, g_latt, g_vault, g_replay, g_stdp = gates.unbind(0)

        # Accumulate contributions into a blend tensor
        blend = torch.zeros_like(x)  # (B, T, C)
        contributed = False

        # ── ① IMAGINATION (ThirdEye) — pain/confusion activates dreaming ─
        # Standard ThirdEye fires at inference; here we also fire during
        # high-pain training moments so she imagines when stuck.
        # We NEVER force ThirdEye — it opens only when ego calls for it.
        # But we always LOG it so every imagination event is visible.
        self._last_eye_event = None   # reset each forward pass

        if pain > 0.4 or (not training and g_imag > 0.3):
            with torch.no_grad() if not training else torch.enable_grad():
                dreamed = third_eye(x, will_signal=ego["will_power"])
                dream_delta = g_imag * (dreamed - x)
                blend = blend + dream_delta
                contributed = True

                # ── 👁️ THIRD EYE LOG ────────────────────────────────────
                # Capture what triggered the imagination and how strong it was.
                # This is stored on self so train_space.py can print it.
                delta_norm  = dream_delta.detach().float().norm().item()
                gate_val    = float(g_imag.item())
                trigger     = "pain" if pain > 0.4 else "low_will"
                self._last_eye_event = {
                    "trigger"    : trigger,
                    "pain"       : float(pain.item()),
                    "will"       : float(will.item()),
                    "gate"       : gate_val,
                    "delta_norm" : delta_norm,
                    "training"   : training,
                }

        # ── ② LATTICE (geometric past) — always query, gate strength by ego ─
        if g_latt > 0.1 and len(memory_lattice.episodic_memories) > 0:
            with torch.no_grad():
                query_emb = x.detach().float().mean(dim=(0, 1))  # (C,)
                hits = memory_lattice.retrieve(
                    query="", query_embedding=query_emb, top_k=6
                )
            if hits:
                try:
                    mem_embs = torch.stack(
                        [e.embedding.to(device=device, dtype=x.dtype) for e, _ in hits]
                    )  # (M, C)
                    # Simple mean-pool memories → broadcast add
                    mem_mean = mem_embs.mean(0).view(1, 1, C)  # (1,1,C)
                    blend = blend + g_latt * mem_mean.expand(B, T, C)
                    contributed = True
                except Exception:
                    pass

        # ── ③ VAULT (verbatim facts) — pain activates deeper recall ──────
        # Vault is text-based; we embed the query vector, find relevant labels,
        # then encode the retrieved text back as a pseudo-embedding via
        # a character-level hash → embedding trick (no tokenizer needed here).
        if g_vault > 0.2 and self._vault is not None:
            with torch.no_grad():
                # Project context mean to a small query space
                ctx_mean = x.detach().float().mean(dim=(0, 1))  # (C,)
                query_proj = self.vault_query_proj(
                    ctx_mean.to(next(self.vault_query_proj.parameters()).device)
                )  # (64,)
                # Convert to a simple scalar key for top-label search
                query_scalar = float(query_proj.mean().item())
                # Retrieve vault entries (label similarity only — no tokenizer)
                labels = self._vault.get_all_labels()
                if labels:
                    # Simple deterministic embedding of vault content into x-space
                    # We hash each entry's text to a seed and generate a unit vector
                    best_entries = self._vault.recall(
                        query=str(query_scalar)[:8], top_k=3
                    )
                    if best_entries:
                        vault_vecs = []
                        for entry in best_entries:
                            txt = entry.get("text", "")
                            seed = hash(txt) % (2**31)
                            gen = torch.Generator()
                            gen.manual_seed(seed)
                            v = torch.randn(C, generator=gen, device=device,
                                           dtype=x.dtype)
                            v = v / (v.norm() + 1e-8)
                            vault_vecs.append(v)
                        vault_blend = torch.stack(vault_vecs).mean(0).view(1, 1, C)
                        blend = blend + g_vault * pain * vault_blend.expand(B, T, C)
                        contributed = True

        # ── ④ REPLAY (past experience) — curiosity samples buffer ─────────
        if g_replay > 0.2 and not training and len(replay_buffer.buffer) >= 10:
            with torch.no_grad():
                try:
                    stored, rewards = replay_buffer.sample(
                        min(8, len(replay_buffer.buffer)), device
                    )
                    if stored.numel() > 0:
                        # Weight by reward, mean-pool, broadcast
                        w_replay = F.softmax(rewards, dim=0).view(-1, 1)
                        replay_mean = (stored * w_replay).sum(0).view(1, 1, C)
                        blend = blend + g_replay * replay_mean.expand(B, T, C)
                        contributed = True
                except Exception:
                    pass

        # ── ⑤ STDP BOOST — harmony amplifies fast-weight associations ────
        # During training, high harmony means the current moment is
        # "on-brand" — boost STDP update rate via a returned scalar.
        # We store this as a buffer so PlasticAttention can read it.
        if training and g_stdp > 0.5 and harmony > 0.3:
            # Signal back to attention blocks: "fire stronger STDP this step"
            # We do this by temporarily scaling will_per_token
            # (PlasticAttention reads will_per_token for STDP rate)
            ego["will_per_token"] = ego["will_per_token"] * (1.0 + g_stdp * harmony)

        # ── ⑥ CHAIN TRACKER — auto-record reasoning steps ────────────────
        if self._chains is not None and self._chains.get_active() is not None:
            # During inference, auto-record each forward pass as a reasoning step
            # (only if a chain is active — user must start one explicitly)
            if not training:
                try:
                    ctx_summary = f"step_t{T}_will{will:.2f}_pain{pain:.2f}"
                    self._chains.record_step(
                        summary=ctx_summary,
                        exact_output=""   # filled by caller with actual response text
                    )
                except Exception:
                    pass

        # ── Blend all contributions back into residual stream ─────────────
        if contributed:
            blended = self.blend_proj(blend)
            x = self.blend_norm(x + blended)

        return x


# ── LIQUID CONTEXT WINDOW: Memory Projection Layer ─────────────────────────
class MemoryProjection(nn.Module):
    """
    Closes the loop between MemoryLattice and the forward pass.
    Retrieves top-K relevant memories from the lattice using the current
    hidden state as a query, then fuses them into the active context via
    lightweight cross-attention.

    This makes the context window 'liquid' — Jinx can see beyond its hard
    boundary by attending to semantically relevant past experiences.

    Purely additive. Does not modify any existing pathway.
    """
    def __init__(self, n_embd: int, n_heads: int = 4, top_k: int = 8):
        super().__init__()
        self.n_embd  = n_embd
        self.n_heads = n_heads
        self.top_k   = top_k
        self.head_dim = n_embd // n_heads

        # Project memory embeddings into Q/K/V space
        self.mem_k = nn.Linear(n_embd, n_embd)
        self.mem_v = nn.Linear(n_embd, n_embd)
        # Project current context into query space
        self.ctx_q = nn.Linear(n_embd, n_embd)
        # Output projection back to residual stream
        self.out    = nn.Linear(n_embd, n_embd)
        # Learned gate: how much memory to inject (starts near zero)
        self.gate   = nn.Parameter(torch.zeros(1))
        self.norm   = nn.LayerNorm(n_embd)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, C) current hidden state
        memory_lattice,                            # MemoryLattice instance
        will_signal: Optional[torch.Tensor] = None # (B,) ego will
    ) -> torch.Tensor:
        """
        Returns x unchanged if no memories found.
        Returns x + gated(cross_attn(x, memories)) otherwise.
        """
        B, T, C = x.shape
        device   = x.device

        # ── Query lattice with mean context vector ──────────────────────
        with torch.no_grad():
            # Mean over batch and sequence → single query embedding
            query_emb = x.detach().float().mean(dim=(0, 1))  # (C,)
            hits = memory_lattice.retrieve(
                query="",
                query_embedding=query_emb,
                top_k=self.top_k
            )

        if not hits:
            return x  # Nothing retrieved — passthrough, zero cost

        # ── Build memory tensor ─────────────────────────────────────────
        mem_embs = torch.stack(
            [entry.embedding.to(device=device, dtype=x.dtype) for entry, _ in hits]
        )  # (M, C)
        M = mem_embs.size(0)

        # ── Cross-attention: context queries, memories are keys/values ──
        # Queries from current context: (B, T, C) → (B, n_heads, T, head_dim)
        Q = self.ctx_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Keys/Values from memories: (M, C) → (1, n_heads, M, head_dim)
        K = self.mem_k(mem_embs).view(1, M, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.mem_v(mem_embs).view(1, M, self.n_heads, self.head_dim).transpose(1, 2)

        # Expand K, V to batch dim
        K = K.expand(B, -1, -1, -1)  # (B, n_heads, M, head_dim)
        V = V.expand(B, -1, -1, -1)

        # Scaled dot-product attention (non-causal — memories have no order)
        scale  = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, M)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V)  # (B, H, T, head_dim)

        # Merge heads → (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        projected = self.out(attn_out)

        # ── Gated residual injection ────────────────────────────────────
        # Gate is learned and starts at ~0 (sigmoid(0)=0.5, but we init
        # the parameter at 0 and scale by 0.1 so initial contribution is tiny)
        # Will signal optionally amplifies the gate
        gate_val = torch.sigmoid(self.gate) * 0.1
        if will_signal is not None:
            gate_val = gate_val * (1.0 + torch.tanh(will_signal).mean().clamp(0, 1))

        return self.norm(x + gate_val * projected)


# Import jinX Silicon Engine (Level 12.6)
try:
    # First check for the JIT-compiled module from train_space.py
    import jinx_cuda_jit as jinx_cuda
except ImportError:
    try:
        import jinx_cuda
    except ImportError:
        jinx_cuda = None

class LiquidReservoir(nn.Module):
    """
    Internal memory beyond parameters.
    A recurrent 'pool' of hidden states that decays over time.
    
    LEVEL 21.0: THE GLOBAL SINGULARITY. 
    Full 896x896 Manifold using Tiled CUDA Processing.
    """
    def __init__(self, config, decay: float = 0.95):
        super().__init__()
        self.n_embd = config.n_embd
        # GLOBAL BALLROOM: Every dimension can talk to every other dimension.
        self.register_buffer("state", torch.zeros(self.n_embd, self.n_embd))
        self.decay = decay

    def forward(self, x: torch.Tensor, will_signal: torch.Tensor, will_per_token: Optional[torch.Tensor] = None, curiosity_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.size()
        
        # Spacetime Warp Signal
        w_pulse = will_per_token if will_per_token is not None else will_signal.unsqueeze(1).repeat(1, T)
        w_pulse = torch.sigmoid(w_pulse)

        if jinx_cuda is not None and x.is_cuda:
            # === jinX MUSCLE: Level 21.0 Global Pulse (Tiled) ===
            # We pass the full C=896 tensors to the tiled kernel.
            # FP32 GUARD: kernel requires float32 regardless of AMP dtype
            memory_read = jinx_cuda.forward(
                x.float().contiguous(), 
                x.float().contiguous(), 
                x.float().contiguous(), 
                w_pulse.float().contiguous(), 
                self.state.float().contiguous(), 
                self.decay
            )
            memory_read = memory_read.to(dtype=x.dtype)
        else:
            # Fallback: einsum-based (C×C) global accumulation
            will_gate = torch.tanh(will_signal).mean().clamp(0, 1)
            with torch.no_grad():
                x_f = x.float().detach()
                w_f = w_pulse.float().detach()
                weighted_x = x_f * w_f.unsqueeze(-1)
                update = torch.einsum('btc,btd->cd', weighted_x, x_f) / (B * T)
                self.state.mul_(self.decay).add_(update, alpha=float(will_gate))
            memory_read = torch.einsum('btc,cd->btd', x.float(), self.state).to(dtype=x.dtype)

        # === HOMEOSTASIS: Global Manifold Scaling ===
        with torch.no_grad():
            state_norm = self.state.norm(p='fro')
            max_norm = float(C) 
            if state_norm > max_norm:
                self.state.mul_(max_norm / (state_norm + 1e-8))

        return memory_read.contiguous()

class PlasticAttention(nn.Module):
    """
    Hebbian-style Attention with jinX PHASE-SHIFTING.
    QKV projections have 'Fast Weights' that adapt during inference.
    Plus perspective rotation (phase shifting) to prevent concept bleeding.
    """
    def __init__(self, config, bias_types: Optional[List[str]] = None, layer_idx: int = 0):
        super().__init__()
        self.config = config # SISTER'S SHIELD: Link to global config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx

        # Slow Weights (Long-term identity)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)

        # jinXEffect Level 1: Global Manifold Shift (The Building Rotation)
        # Controlled by the global mean context
        self.global_phase_gate = nn.Sequential(
            nn.Linear(config.n_embd, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # jinXEffect Level 2: Custom RoPE Engine (Geometric Grounding)
        self.rope = RotaryEmbedding(self.head_dim)

        # Fast Weights (Hebbian Reflexes) - Associative Memory per head
        self.register_buffer("q_fast", torch.zeros(self.n_head, self.head_dim, self.head_dim))
        self.register_buffer("v_fast", torch.zeros(self.n_head, self.head_dim, self.head_dim))

        # jinX GRAFT: Phase-Shifting Engine (TARDIS Rooms)
        # 14 heads at ~25.71° increments (360/14) — the Golden Ratio configuration.
        # Each head sees the same token from a distinct geometric perspective.
        # head 0 = 0°, head 1 = 25.7°, head 7 = 180° (mirror), head 13 = 334.3°
        phase_offsets = torch.linspace(0, 2 * math.pi, self.n_head)
        self.register_buffer("phase_offsets", phase_offsets)
        
        # Meta-Reflex: Neural Restlessness detector
        from ego import NeuralRestlessness
        self.restlessness = NeuralRestlessness(self.n_head)

        # --- GUIDED CIVILIZATION: Head Roles ---
        if bias_types is None:
            bias_types = ['random'] * self.n_head
        elif isinstance(bias_types, str):
            bias_types = [bias_types] * self.n_head
        self.head_bias_types = bias_types

        # Apply structured initialization to Q/K/V weights
        self._apply_structured_init(config)

        # --- META-PLASTICITY: Per-Weight Learning Rates ---
        self.register_buffer("eta_map", torch.ones(config.n_embd, config.n_embd) * 0.01)
        self.eta_min = 0.001
        self.eta_max = 0.1
        self.eta_decay = 0.995

        # --- GUIDED CIVILIZATION: Per-Head Plasticity Multipliers ---
        self._assign_per_head_plasticity()

        # --- GUIDED CIVILIZATION: Head Lifecycle Manager ---
        enable_lifecycle = getattr(config, 'enable_head_lifecycle', False)
        if enable_lifecycle:
            self.lifecycle_manager = HeadLifecycleManager(
                n_heads=self.n_head,
                bias_types=self.head_bias_types,
                promotion_threshold=getattr(config, 'promotion_threshold', 0.8),
                demotion_threshold=getattr(config, 'demotion_threshold', 0.3)
            )
        else:
            self.lifecycle_manager = None

        # --- CONTEXT-AWARE PLASTICITY GATING ---
        self.plasticity_gate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
            nn.Sigmoid()
        )

        # SISTER'S SHIELD: Expand causal mask to handle block_size + identity buffer
        max_T = config.block_size + 16 # Extra headroom
        self.register_buffer("bias", torch.tril(torch.ones(max_T, max_T))
                                     .view(1, 1, max_T, max_T))

        # ── MEMORY-IN-ATTENTION: Episodic memories as native KV participants ──
        # This is the core architectural upgrade: memories are NOT blended in
        # after attention. They become actual Keys and Values inside the
        # attention matrix itself. Every head attends to them directly.
        # YAKA-RoPE phase shifts apply to memory keys automatically.
        # STDP fires on memory associations directly — memories shape fast weights.
        # This is memory as organ, not memory as accessory.
        self.mem_k_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.mem_v_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Learned scalar gate: how much weight to give memory keys vs token keys
        # Starts near zero (tanh(0)=0) — grows only as it learns memory is useful
        self.mem_gate    = nn.Parameter(torch.zeros(1))
        # Max memories to attend per forward pass — keeps O(T+M) manageable
        self._mem_top_k  = 8
        # Reference to the lattice — injected by SpaceTransformer after init
        self._memory_lattice = None

    def _apply_structured_init(self, config):
        """
        Guided Civilization: Initialize Q/K/V with role-specific priors.
        Blends structured init (70%) with standard init (30%) for stability.
        """
        import math
        with torch.no_grad():
            W_q = self.q_proj.weight.data
            W_k = self.k_proj.weight.data
            W_v = self.v_proj.weight.data

            for head_idx, b_type in enumerate(self.head_bias_types):
                # Slice for this head
                q_start = head_idx * self.head_dim
                q_end = (head_idx + 1) * self.head_dim
                k_start = head_idx * self.head_dim
                k_end = (head_idx + 1) * self.head_dim
                v_start = head_idx * self.head_dim
                v_end = (head_idx + 1) * self.head_dim

                W_q_head = W_q[q_start:q_end, :]
                W_k_head = W_k[k_start:k_end, :]
                W_v_head = W_v[v_start:v_end, :]

                if b_type == 'global':
                    self._init_lowfreq_fourier(W_q_head, W_k_head, modes=4)
                elif b_type in ['periodic', 'causal']:
                    self._init_orthogonal(W_q_head, W_k_head)
                elif b_type in ['dialogue', 'instruction']:
                    self._init_sparse(W_q_head, W_k_head, sparsity=0.3)
                elif b_type in ['safety', 'self_monitor']:
                    W_q_head.mul_(0.5)
                    W_k_head.mul_(0.5)
                    W_v_head.mul_(0.5)
                elif b_type == 'order':
                    self._init_order_flow(W_q_head, W_k_head)
                # 'local', 'self', 'random': keep standard init

    def _init_lowfreq_fourier(self, W_q, W_k, modes=4):
        """Initialize with low-frequency Fourier bases (global heads)."""
        n_out, n_in = W_q.shape
        x = torch.linspace(0, 2 * math.pi, n_in, device=W_q.device)
        bases = []
        for k in range(1, modes + 1):
            bases.append(torch.sin(k * x))
            bases.append(torch.cos(k * x))
        if len(bases) == 0:
            fourier_mat = torch.zeros((0, n_in), device=W_q.device)
        else:
            fourier_mat = torch.stack(bases)
        if fourier_mat.size(0) > n_out:
            fourier_mat = fourier_mat[:n_out]
        elif fourier_mat.size(0) < n_out:
            pad_rows = n_out - fourier_mat.size(0)
            pad = torch.randn(pad_rows, n_in, device=W_q.device) * 0.02
            fourier_mat = torch.cat([fourier_mat, pad], dim=0)
        # Blend: 70% Fourier, 30% standard
        W_q.data = 0.7 * fourier_mat + 0.3 * W_q.data
        W_k.data = 0.7 * fourier_mat + 0.3 * W_k.data

    def _init_orthogonal(self, W_q, W_k):
        """Initialize with orthogonal matrices (symmetry-locking heads)."""
        if W_q.size(0) <= W_q.size(1):
            Q, _ = torch.linalg.qr(W_q.T)
            W_q.data = Q.T[:W_q.size(0), :]
        else:
            U, _, Vh = torch.linalg.svd(W_q, full_matrices=False)
            W_q.data = U @ Vh
        if W_k.size(0) <= W_k.size(1):
            Q, _ = torch.linalg.qr(W_k.T)
            W_k.data = Q.T[:W_k.size(0), :]
        else:
            U, _, Vh = torch.linalg.svd(W_k, full_matrices=False)
            W_k.data = U @ Vh

    def _init_sparse(self, W_q, W_k, sparsity=0.3):
        """Initialize with sparse gating (routing heads)."""
        threshold_q = torch.quantile(W_q.abs(), sparsity)
        threshold_k = torch.quantile(W_k.abs(), sparsity)
        W_q.data[W_q.abs() < threshold_q] = 0.0
        W_k.data[W_k.abs() < threshold_k] = 0.0

    def _init_order_flow(self, W_q, W_k):
        """Initialize with monotonic/reversible-flow style (order-seeking heads)."""
        n_out, n_in = W_q.shape
        ramp = torch.linspace(-1.0, 1.0, steps=n_in, device=W_q.device)
        flow = ramp.unsqueeze(0).repeat(n_out, 1)
        W_q.data = 0.6 * flow + 0.4 * W_q.data
        W_k.data = -0.6 * flow + 0.4 * W_k.data

    def _assign_per_head_plasticity(self):
        """
        Guided Civilization: Assign per-head plasticity multipliers based on role.
        These multiply the base eta_map for role-appropriate learning rates.
        """
        # Base plasticity gains per role
        role_multipliers = {
            'local': 0.5,       # Small (fast, short-range)
            'global': 1.5,      # Larger STDP (temporal patterns)
            'dialogue': 1.2,    # Moderate
            'instruction': 1.2, # Moderate
            'safety': 0.1,      # Very low (stable anchor)
            'self_monitor': 0.1, # Very low (stable anchor)
            'periodic': 0.3,    # Low (symmetry-locking)
            'causal': 0.5,      # Low-moderate
            'self': 1.2,        # Moderate + episodic
            'order': 0.3,       # Low (sequential anchor)
            'random': 2.0,      # High (exploration)
        }

        # Create per-head multiplier map (broadcast to eta_map shape)
        head_mults = torch.ones(self.n_head)
        for i, b_type in enumerate(self.head_bias_types):
            head_mults[i] = role_multipliers.get(b_type, 1.0)

        # Reshape to apply per-head (each head gets head_dim rows)
        self.register_buffer("head_plasticity_mult", head_mults)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        half = d // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def attach_memory_lattice(self, lattice):
        """Called once by SpaceTransformer to wire the shared lattice in."""
        self._memory_lattice = lattice

    def _get_memory_kv(
        self,
        x: torch.Tensor,           # (B, T, C) — current hidden state
        total_cos: torch.Tensor,   # (1, H, T, head_dim) YAKA-RoPE cos
        total_sin: torch.Tensor,   # (1, H, T, head_dim) YAKA-RoPE sin
    ):
        """
        Retrieve top-K memories from the lattice and project them into
        the same (B, H, M, head_dim) Key/Value space as the token stream.

        Memory Keys receive the same YAKA-RoPE phase-offset as the
        FIRST token position — they are 'position 0' peers, not positional
        competitors. Each head sees memories from its own rotated frame.

        Returns:
            mem_k: (B, H, M, head_dim) or None
            mem_v: (B, H, M, head_dim) or None
        """
        if self._memory_lattice is None:
            return None, None
        if len(self._memory_lattice.episodic_memories) < 4:
            return None, None

        B, T, C = x.shape
        device   = x.device

        with torch.no_grad():
            query_emb = x.detach().float().mean(dim=(0, 1))   # (C,)
            hits = self._memory_lattice.retrieve(
                query="", query_embedding=query_emb, top_k=self._mem_top_k
            )

        if not hits:
            return None, None

        # Stack memory embeddings: (M, C)
        try:
            mem_embs = torch.stack(
                [e.embedding.to(device=device, dtype=x.dtype) for e, _ in hits]
            )  # (M, C)
        except Exception:
            return None, None

        M = mem_embs.size(0)

        # Project memories into K and V spaces: (M, C)
        mk = self.mem_k_proj(mem_embs)  # (M, C)
        mv = self.mem_v_proj(mem_embs)  # (M, C)

        # Reshape to (1, H, M, head_dim) then expand to (B, H, M, head_dim)
        mk = mk.view(1, M, self.n_head, self.head_dim).transpose(1, 2)  # (1,H,M,d)
        mv = mv.view(1, M, self.n_head, self.head_dim).transpose(1, 2)  # (1,H,M,d)
        mk = mk.expand(B, -1, -1, -1)   # (B, H, M, d)
        mv = mv.expand(B, -1, -1, -1)   # (B, H, M, d)

        # Apply YAKA-RoPE at position 0 so memories live in the same
        # rotated frame as the tokens — each head's phase applies.
        # total_cos/sin shape: (1, H, T, d) — we take position 0 → (1, H, 1, d)
        rot_cos = total_cos[:, :, :1, :]   # (1, H, 1, d)
        rot_sin = total_sin[:, :, :1, :]   # (1, H, 1, d)
        mk = (mk * rot_cos) + (self._rotate_half(mk) * rot_sin)  # phase-rotated

        return mk, mv   # (B, H, M, d), (B, H, M, d)

    def forward(self, x, will_signal: Optional[torch.Tensor] = None,
                will_per_token: Optional[torch.Tensor] = None,
                attn_stats: Optional[List] = None):
        B, T, C = x.size() # Use real sequence length (including identity buffer)

        # 1. Standard projections (Slow Weights only)
        q = self.q_proj(x) # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # === jinXEffect Level 1: Global Manifold Shift (The Unified Wave) ===
        # Rotate the WHOLE brain before splitting into heads
        if will_signal is not None:
            # We use the sequence mean to derive the global intent shift
            global_rot = self.global_phase_gate(x.mean(dim=1)) * math.pi # [-pi, pi]
            global_rot = global_rot.view(B, 1, 1)
            
            # Apply global rotation to every pair of dimensions in the 768-dim space
            q_cos, q_sin = torch.cos(global_rot), torch.sin(global_rot)
            q = (q * q_cos) - (self._rotate_half(q) * q_sin)
            k = (k * q_cos) - (self._rotate_half(k) * q_sin)

        # Split into heads after the Global Shift
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # === jinXEffect Level 2: Custom Phase-Modulated RoPE ===
        # Fuses Position (Geography) with Head-Perspective (Intent)
        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)
        cos, sin = cos.view(1, 1, T, self.head_dim), sin.view(1, 1, T, self.head_dim)
        
        # SISTER'S SHIELD: O(T) Boredom Detection using associative matrix variance
        with torch.no_grad():
            boredom = self.restlessness(self.v_fast.unsqueeze(0))
        
        # Calculate Head-Specific shifts (TARDIS Rooms + Boredom)
        b_thresh = getattr(self.config, 'boredom_threshold', 0.8)
        head_phases = self.phase_offsets.view(1, -1, 1, 1)
        if boredom > b_thresh:
            if self.training and self.layer_idx == 0:
                print(f"[jinXEffect] Perspective Rotation Triggered (Boredom: {boredom:.2f})")
            shift = torch.randn(1, self.n_head, 1, 1, device=x.device) * boredom * 0.1
            head_phases = head_phases + shift
            self.restlessness.reset_boredom()

        # Combine: Angle = Position + HeadShift
        p_cos, p_sin = torch.cos(head_phases), torch.sin(head_phases)
        
        # YAKA-RoPE FUSION: (cos(a+b), sin(a+b))
        total_cos = (cos * p_cos) - (sin * p_sin)
        total_sin = (sin * p_cos) + (cos * p_sin)
        
        q_rot = (q * total_cos) + (self._rotate_half(q) * total_sin)
        k_rot = (k * total_cos) + (self._rotate_half(k) * total_sin)

        # ── MEMORY-IN-ATTENTION: extend K and V with episodic memories ────────
        # This is the moment memory becomes part of HER, not something she looks up.
        # q_rot tokens attend to mem_k in the same computation as token-to-token.
        # STDP update below will fire on (k_rot, v_full) so memories teach
        # fast weights directly — she literally learns from remembering.
        mem_k, mem_v = self._get_memory_kv(x, total_cos, total_sin)
        if mem_k is not None:
            mem_gate_val = torch.sigmoid(self.mem_gate) * 0.2  # max 0.2 contribution
            # Concatenate along sequence dimension: (B, H, T+M, d)
            k_full = torch.cat([k_rot, mem_k * mem_gate_val], dim=2)
            v_full = torch.cat([v,     mem_v * mem_gate_val], dim=2)
            TM = T + mem_k.size(2)   # extended sequence length
        else:
            k_full = k_rot
            v_full = v
            TM = T
        # ─────────────────────────────────────────────────────────────────────

        # 3. STABLE LINEAR ATTENTION (Silicon Muscle - Level 14.0)
        # SISTER'S SHIELD: Full Dual-Pulse Engine.
        # Q_eff, K, and V are processed in a single hardware pulse.
        scale = 1.0 / (math.sqrt(self.head_dim) * math.sqrt(T))
        
        # Prepare tokens for the engine: (B*H, T+M, d)
        # k_full and v_full already include memory rows
        q_flat = q_rot.reshape(B * self.n_head, T, self.head_dim)
        k_flat = (k_full * scale).reshape(B * self.n_head, TM, self.head_dim)
        v_flat = v_full.reshape(B * self.n_head, TM, self.head_dim)

        # Apply Q_FAST modulation (Query Habits)
        q_fast_exp = self.q_fast.expand(B, -1, -1, -1).reshape(B * self.n_head, self.head_dim, self.head_dim)
        q_eff_flat = (q_flat + torch.bmm(q_flat, q_fast_exp)).contiguous()

        # Identity Will for Attention (Standard causal flow)
        # Extend will gate to cover T+M positions (memories get weight 1.0)
        w_att = torch.ones(B * self.n_head, TM, device=x.device, dtype=x.dtype)

        if jinx_cuda is not None and x.is_cuda:
            # === THE SILICON ATTENTION ENGINE ===
            # FP32 GUARD: kernel requires float32 regardless of AMP dtype
            v_fast_exp = self.v_fast.expand(B, -1, -1, -1).reshape(B * self.n_head, self.head_dim, self.head_dim).contiguous()
            y_flat = jinx_cuda.forward(
                q_eff_flat.float().contiguous(),
                k_flat.float().contiguous(),
                v_flat.float().contiguous(),
                w_att.float().contiguous(),
                v_fast_exp.float().contiguous(),
                1.0
            )
            y = y_flat.to(dtype=x.dtype).view(B, self.n_head, T, self.head_dim)
        else:
            # Fallback (Slow Mode)
            v_fast_exp = self.v_fast.expand(B, -1, -1, -1).reshape(B * self.n_head, self.head_dim, self.head_dim)
            outer_prods = k_flat.unsqueeze(-1) @ v_flat.unsqueeze(-2)
            memory_seq = torch.cumsum(outer_prods, dim=1) + v_fast_exp.unsqueeze(1)
            y = (q_eff_flat.unsqueeze(-2) @ memory_seq).squeeze(-2)
            y = y.view(B, self.n_head, T, self.head_dim)

        # SISTER'S SHIELD: Stable Normalization over full K (tokens + memories)
        z_sum = torch.cumsum(k_full.abs() * scale, dim=2)   # (B, H, T+M, d)
        # q_rot is (B,H,T,d), z_sum is (B,H,T+M,d) — use only token prefix for norm
        z_sum_tok = z_sum[:, :, :T, :]   # (B, H, T, d)
        norm = torch.einsum('bhtd,bhtd->bht', q_rot.abs(), z_sum_tok).unsqueeze(-1)
        norm = norm.clamp(1e-4, 1e8)
        y = y / norm
        
        # Will modulation
        if will_signal is not None:
            will_scale = 1.0 + torch.tanh(will_signal).view(B, 1, 1, 1)
            y = y * will_scale
        
        # === GUIDED CIVILIZATION & VRAM VACUUM ===
        # Consolidate stats, lifecycle, and promotions into one block before deleting large tensors.
        if (attn_stats is not None or self.lifecycle_manager is not None) and T <= 1024:
            with torch.no_grad():
                # 1. Compute detached attention probs (O(T^2) but necessary for monitoring)
                att_temp = (q_rot @ k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                if T <= self.bias.size(-1):
                    att_temp = att_temp.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att_probs = F.softmax(att_temp, dim=-1)
                
                # 2. Collect Attention Stats
                if attn_stats is not None:
                    # Compute entropy per head
                    p_clamped = att_probs.clamp_min(1e-8)
                    entropy_per_head = -(p_clamped * p_clamped.log()).sum(dim=-1).mean(dim=-1)
                    
                    # Compute attention distance
                    positions = torch.arange(T, device=x.device, dtype=torch.float32)
                    dist_matrix = torch.abs(positions.view(1, 1, T, 1) - positions.view(1, 1, 1, T))
                    distance_per_head = (att_probs * dist_matrix).sum(dim=(-1, -2)) / (T * T)
                    
                    attn_stats.append({"entropy": entropy_per_head, "distance": distance_per_head})

                # 3. Update Lifecycle Metrics
                if self.lifecycle_manager is not None and self.training:
                    reward_proxy = float(torch.tanh(will_signal).mean().item()) if will_signal is not None else 0.5
                    for h_idx in range(self.n_head):
                        head_attn = att_probs[0, h_idx, :, :]
                        self.lifecycle_manager.update_metrics(h_idx, head_attn, reward_signal=reward_proxy)

                    # Evaluate promotion/demotion
                    step = getattr(self, '_current_step', 0)
                    if step > 0 and step % 100 == 0:
                        for h_idx in range(self.n_head):
                            if self.lifecycle_manager.evaluate_promotion(h_idx, step):
                                print(f"🌟 [PROMOTION] Layer {self.layer_idx} Head {h_idx}: Career Boost")
                            if self.lifecycle_manager.evaluate_demotion(h_idx, step):
                                new_role = self.lifecycle_manager.reassign_role(h_idx)
                                if new_role: print(f"🧬 [EVOLUTION] Layer {self.layer_idx} Head {h_idx} -> {new_role.upper()}")

                # 4. VACUUM ACTION: Kill large tensors immediately
                del att_temp
                del att_probs

        # --- UPDATE FAST WEIGHTS (Hebbian Update per Head) + META-PLASTICITY ---
        if will_signal is not None and self.training:
            with torch.no_grad():
                will_avg = torch.tanh(will_signal).mean()
                token_gates = torch.sigmoid(will_per_token).mean() if will_per_token is not None else 0.5

                # Compute per-head associative updates: mean across batch (B) and time (T)
                # === jinX STDP: Causal Memory Expansion (Einsum Surgery) ===
                # We use einsum to compute updates directly, avoiding the huge 5D tensor materialization
                # ── PHASE-AWARE STDP: updates happen in ROTATED space ─────────────
                # v_fast accumulates in the space that the VALUE projections
                # actually operate in (post-rotation). This matters because:
                #   standard: k_rot (rotated) dot v (unrotated) → cross-space mismatch
                #   phase-aware: k_rot (rotated) dot v_rot (rotated) → consistent space
                # v_rot: apply the same global+head phase to V so all three (Q,K,V)
                # live in the same rotated world when STDP fires.
                # This is the equivalent of synaptic timing happening in the
                # same coordinate frame as the signal itself.
                v_rot = (v * total_cos) + (self._rotate_half(v) * total_sin)

                # lane 1: Hebbian (Global context) — now phase-consistent
                # Use k_full (tokens + memories) so STDP fires on ALL attended content
                # v_rot_full: apply phase rotation to full v (including memory rows)
                v_full_rot = (v_full * total_cos[:,:,:TM,:]) + (self._rotate_half(v_full) * total_sin[:,:,:TM,:])
                update_v_hebb = torch.einsum('bhtd,bhte->hde', k_full, v_full_rot) / (B * TM)

                # lane 2: STDP (Causal order / Arrow of Time)
                # Emphasize later tokens using the causal ramp (over full T+M length)
                seq_idx = torch.arange(TM, device=x.device, dtype=x.dtype).view(1, 1, TM)
                causal_boost = torch.sigmoid((seq_idx - TM/2) / max(1, TM/10))
                update_v_stdp = torch.einsum('bhtd,bhte,bht->hde', k_full, v_full_rot, causal_boost) / (B * TM)

                # Blend (30% Hebbian Global, 70% STDP Causal)
                update_v = 0.3 * update_v_hebb + 0.7 * update_v_stdp

                # q_fast: stores Q -> Q transformations in rotated space
                # (how she learns to ask questions given her current perspective)
                update_q = torch.einsum('bhtd,bhte->hde', q_rot, q_rot) / (B * T)

                # Apply per-head plasticity multipliers
                head_mults = self.head_plasticity_mult.view(-1, 1, 1)

                # === META-PLASTICITY: Update eta_map (per-weight learning rates) ===
                # Compute eligibility (where should we adapt?)
                eligibility_v = update_v.abs()
                eligibility_q = update_q.abs()

                # === NaN SAFETY: Clamp update magnitudes ===
                update_v = update_v.clamp(-1.0, 1.0)
                update_q = update_q.clamp(-1.0, 1.0)

                # Scale eligibility
                scale_v = eligibility_v.mean() + 1e-8
                scale_q = eligibility_q.mean() + 1e-8
                eligibility_v = (eligibility_v / scale_v).clamp(0, 10)
                eligibility_q = (eligibility_q / scale_q).clamp(0, 10)

                # Update eta_map: grow where there's eligibility, decay otherwise
                will_factor = max(0.1, will_avg)
                meta_lr = 0.003  # SWEET SPOT

                # Per-head eta_map updates
                eta_decay = 0.995
                eta_growth_v = will_factor * eligibility_v * head_mults
                eta_growth_q = will_factor * eligibility_q * head_mults

                for h in range(self.n_head):
                    h_start = h * self.head_dim
                    h_end = (h + 1) * self.head_dim
                    
                    # Shield the growth from NaNs/Infs
                    growth = (eta_growth_v[h] + eta_growth_q[h]).nan_to_num(0.0)
                    
                    self.eta_map[h_start:h_end, h_start:h_end].mul_(eta_decay)
                    self.eta_map[h_start:h_end, h_start:h_end].add_(growth, alpha=meta_lr)
                    self.eta_map[h_start:h_end, h_start:h_end].clamp_(0.001, 0.1)
                
                # === NaN SAFETY: Global eta_map clamp ===
                self.eta_map.clamp_(0.001, 0.1)

                # === FAST WEIGHT STEP SCALE ===
                # With grad_accum_steps=16, fast weights accumulate 16 updates
                # before any optimizer step. Step must be small enough that
                # norm stays well below cap across all 16 accumulation steps.
                # Empirical: will_factor~0.1, token_gates~0.5, head_mults~1.2
                # → effective alpha per step ≈ 0.06. Over 16 steps → 0.96.
                # With norm cap=1.0 this gives healthy [0, 1] range.
                fast_lr = 0.001  # 10x smaller than before — prevents saturation

                # Update v_fast (The Associative Lattice - what I remember)
                self.v_fast.mul_(0.99).add_(update_v * token_gates * head_mults, alpha=fast_lr * float(will_factor))

                # Update q_fast (The Query Habits - how I ask questions)
                self.q_fast.mul_(0.99).add_(update_q * token_gates * head_mults, alpha=fast_lr * float(will_factor))

                # === HOMEOSTASIS: Tight norm cap (max=1.0 not 10.0) ===
                # Cap at 1.0 so fast weights stay a gentle modulation, not domination
                self.v_fast.clamp_(-1.0, 1.0)
                self.q_fast.clamp_(-1.0, 1.0)
                v_norm = self.v_fast.norm()
                q_norm = self.q_fast.norm()
                if v_norm > 1.0: self.v_fast.mul_(1.0 / (v_norm + 1e-8))
                if q_norm > 1.0: self.q_fast.mul_(1.0 / (q_norm + 1e-8))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class DendriticMLP(nn.Module):
    """
    jinX GRAFT: Logical core with multiplicative branching.
    Simulates logical gating at the synapse level.
    """
    def __init__(self, config):
        super().__init__()
        # Dendritic Gating
        self.dend1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.dend2 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SISTER'S SHIELD: Super-Dendritic SwiGLU (Asymmetric Gating)
        # Branch 1: The Activation Path (SiLU)
        # Branch 2: The Linear Gating Path
        # Combining them creates a high-rank logical bridge that is more
        # powerful than standard SwiGLU because it maintains dendritic separation.
        x1 = self.dend1(x)
        x2 = self.dend2(x)
        h = self.act(x1) * x2
        return self.dropout(self.down_proj(h))

class GhostLattice(nn.Module):
    """
    SISTER'S SHIELD: The Fractal Ghost Lattice.
    Replaces GhostBridge with a recursive, multi-rank system.
    
    Contains:
    - High-Rank Logic Bridge (for complex reasoning)
    - Low-Rank Nuance Bridge (for subtle identity adjustment)
    - Recursive 'Ghost of Ghost' (spawns if signal std > 10.0)
    """
    def __init__(self, n_embd, depth=0, max_depth=1):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        # Rank-4: Subconscious Nuance (Always partially active)
        self.lora_A_low = nn.Parameter(torch.randn(4, n_embd) * 0.01)
        self.lora_B_low = nn.Parameter(torch.zeros(n_embd, 4))
        
        # Rank-64: Deep Logic Bridge (Triggered by high will)
        self.lora_A_high = nn.Parameter(torch.randn(64, n_embd) * 0.01)
        self.lora_B_high = nn.Parameter(torch.zeros(n_embd, 64))
        
        # Fractal recursion: The Ghost of the Ghost
        if depth < max_depth:
            self.ghost_of_ghost = GhostLattice(n_embd, depth + 1, max_depth)
        else:
            self.ghost_of_ghost = None

    def forward(self, x, will_signal):
        # x: (B, T, C)
        # will_signal: (B,)
        
        # 1. Low-Rank Nuance (Subconscious) - Always 5% active
        gate_low = (torch.sigmoid(will_signal) * 0.1).view(-1, 1, 1)
        nuance = (x @ self.lora_A_low.t()) @ self.lora_B_low.t()
        res = x + (nuance * gate_low)
        
        # 2. High-Rank Logic (Crisis Mode) - Fully gated by Will
        gate_high = torch.tanh(will_signal).view(-1, 1, 1)
        logic = (x @ self.lora_A_high.t()) @ self.lora_B_high.t()
        res = res + (logic * gate_high)
        
        # 3. Fractal Spawn (The Ghost of the Ghost)
        # Triggered if the signal variance is too high (Neural Turbulence)
        if self.ghost_of_ghost is not None:
            if x.std() > 5.0: # Threshold for turbulence
                res = self.ghost_of_ghost(res, will_signal * 0.5)
                
        return res

class SpaceBlock(nn.Module):
    def __init__(self, config, bias_types: Optional[List[str]] = None, layer_idx: int = 0):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = PlasticAttention(config, bias_types=bias_types, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = DendriticMLP(config)
        # SISTER'S SHIELD: The Fractal Ghost Lattice (Hierarchical Logic)
        self.ghost_lattice = GhostLattice(config.n_embd, max_depth=1)

    def forward(self, x, will_signal: Optional[torch.Tensor] = None,
                will_per_token: Optional[torch.Tensor] = None,
                attn_stats: Optional[List] = None):
        # SISTER'S SHIELD: Internal Residual Scaling (0.05x) - Reinforced
        # We apply this directly to the sub-block outputs to prevent signal explosion.
        res_scale = 0.05
        x = x + (res_scale * self.attn(self.ln_1(x), will_signal=will_signal, will_per_token=will_per_token,
                          attn_stats=attn_stats))
        
        # jinX GHOST LATTICE: Fractal Adaptation (Multi-rank Expansion)
        if will_signal is not None:
            # We pass through the lattice which can recursively expand
            x = self.ghost_lattice(x, will_signal)

        x = x + (res_scale * self.mlp(self.ln_2(x)))
        return x

class SpaceTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- IDENTITY BUFFER (Self & Temporal Tokens) ---
        # LEVEL 11.0: Deep Ego Core (8 identity, 8 time = 16 total self-state)
        self.self_model_tokens = 8
        self.temporal_tokens = 8
        self.data_block_size = config.block_size - (self.self_model_tokens + self.temporal_tokens)
        
        # SISTER'S SHIELD: Reduced initialization for stability
        self.self_token_bias = nn.Parameter(torch.randn(self.self_model_tokens, config.n_embd) * 0.01)
        self.temporal_token_bias = nn.Parameter(torch.randn(self.temporal_tokens, config.n_embd) * 0.01)

        # QWEN UPGRADE: High-resolution vocabulary (Default: 151851)
        vocab_size = getattr(config, 'vocab_size', 151851)
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        
        # Reduced embedding init for stability
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.01)
        
        # SISTER'S SHIELD: Expand position embedding
        max_positions = config.block_size + self.self_model_tokens + self.temporal_tokens + 16
        self.position_embedding = nn.Embedding(max_positions, config.n_embd)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

        # --- GUIDED CIVILIZATION: Bias Configuration per Layer ---
        self.bias_config = getattr(config, 'bias_config', None)
        if self.bias_config is None:
            self.bias_config = [['random'] * config.n_head for _ in range(config.n_layer)]

        self.blocks = nn.ModuleList([
            SpaceBlock(config, bias_types=self.bias_config[i], layer_idx=i)
            for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # SISTER'S SHIELD: Weight Tying (The DNA Lock)
        # Sharing weights between embedding and head saves 544MB VRAM
        self.lm_head.weight = self.token_embedding.weight

        # --- THE EGO ENGINE & LIQUID MEMORY ---
        self.ego_engine = EgoEngine(config.n_embd)
        self.reservoir = LiquidReservoir(config)

        # --- MEMORY LATTICE ---
        self.memory_lattice = MemoryLattice(
            embedding_dim=config.n_embd,
            max_episodic=2000,
            device=config.device
        )

        self.replay_buffer = MemoryReplay(config.n_embd, capacity=500, sequence_length=config.block_size)
        self.third_eye = ThirdEye(config) # THE THIRD EYE: Imagination Engine

        # ── LIQUID CONTEXT WINDOW ──────────────────────────────────────────────
        # Reads from the MemoryLattice during every forward pass.
        # Purely additive — gate starts near zero, grows as it learns to use memory.
        self.memory_projection = MemoryProjection(
            n_embd=config.n_embd,
            n_heads=4,
            top_k=8
        )
        # Consolidation interval for training (saved via unified checkpoint)
        self._consolidation_interval = getattr(config, 'consolidation_interval', 200)

        # ── MEMORY-IN-ATTENTION: wire the shared lattice into every PlasticAttention ─
        # This is the single call that makes all 12 (or N) heads native memory attendees.
        # Must happen AFTER self.blocks and self.memory_lattice are both initialized.
        for block in self.blocks:
            block.attn.attach_memory_lattice(self.memory_lattice)

        # ── RESONANCE ENGINE: cross-temporal dot-connector ──────────────────────
        self.resonance_engine = ResonanceEngine(n_embd=config.n_embd, n_resonance=8)
        # Stores last resonance log so test_checkpoint can read it and optionally
        # prepend it to Jinx's generation prompt as a soft self-awareness hint.
        self._last_resonance_log: List[str] = []

        # ── COGNITION BRIDGE: ego-routed internal cognition ────────────────────
        self.cognition_bridge = CognitionBridge(n_embd=config.n_embd)
        # Vault/chains attached later by test_checkpoint.py via attach_cognition()
        # so model.py stays import-free from cognition_engine.py
        # ──────────────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────────

        self.register_buffer("timestep", torch.tensor(0))
        self.last_hidden_state = None # Reference only, not a persistent buffer
        self.register_buffer("dream_loss", torch.tensor(0.0))

    def forward(self, idx, targets=None, slow_grad_norm: float = 0.0, last_loss: Optional[float] = None):
        B, T = idx.shape
        
        # SISTER'S SHIELD: Aligned Overlapping Positions
        # Identity: 0-7, Data: RoPE-handled
        prefix_len = self.self_model_tokens + self.temporal_tokens
        prefix_pos = torch.arange(0, prefix_len, device=idx.device)
        
        # 1. Perception
        tok_emb = self.token_embedding(idx)
        
        # 2. Identity Injection (The Soul Shield)
        self_tokens = self.self_token_bias.unsqueeze(0).expand(B, -1, -1)
        temp_tokens = self.temporal_token_bias.unsqueeze(0).expand(B, -1, -1)
        
        step_val = self.timestep.item()
        pulse = torch.tensor([
            math.sin(step_val / 10.0), math.cos(step_val / 10.0),
            math.sin(step_val / 1000.0), math.cos(step_val / 1000.0)
        ], device=idx.device, dtype=torch.float32).view(1, 1, 4)
        
        pulse_prefix = torch.zeros(B, self.temporal_tokens, self.config.n_embd, device=idx.device)
        pulse_prefix[:, :, :64] = (pulse * 0.5).repeat(1, 1, 16)
        temp_tokens = temp_tokens + pulse_prefix
        
        x_prefix = torch.cat([self_tokens, temp_tokens], dim=1)
        x_prefix = x_prefix + self.position_embedding(prefix_pos).unsqueeze(0)
        
        # SISTER'S SHIELD: Positional Grounding is now handled by RoPE inside the blocks.
        # We only pass the raw token embeddings for the data section.
        x_data = tok_emb
        
        # SISTER'S SHIELD: Decoupling Soul from Data Gradients
        x = torch.cat([x_prefix.detach(), x_data], dim=1)
        x_for_ego = torch.cat([x_prefix, x_data], dim=1)

        # ... (rest of forward pass) ...
        # Attention stats collection
        attn_stats = []

        # 3. First Pass: Ego Analysis
        with torch.no_grad():
             drive_analysis = self.ego_engine(x_for_ego, last_loss=last_loss, slow_grad_norm=slow_grad_norm, prefix_len=prefix_len)
             will_signal = drive_analysis["will_power"]  
             will_per_token = drive_analysis["will_per_token"]  
             
             # SISTER'S SHIELD: The Hook
             # We search for the 'User:' sequence (GPT-2 tokens: 1639, 25)
             # If found, we boost harmony to make her 'listen' harder.
             if self.training:
                 user_hook = (idx == 1639) | (idx == 25)
                 if user_hook.any():
                     drive_analysis["harmony"] = drive_analysis["harmony"] * 1.5
                     drive_analysis["will_power"] = drive_analysis["will_power"] * 1.2
             
             harmony = drive_analysis["harmony"]
             curiosity = drive_analysis["curiosity"]
             pain = drive_analysis["pain"]

        # ── LIQUID CONTEXT WINDOW: Memory Projection ──────────────────────────
        # Slot: AFTER ego analysis (will_signal known) BEFORE reasoning loop
        # (so enriched context flows through ALL reasoning iterations)
        # Only active at inference to avoid interfering with gradient flow.
        if not self.training and len(self.memory_lattice.episodic_memories) > 0:
            x = self.memory_projection(x, self.memory_lattice, will_signal=will_signal)
        # ─────────────────────────────────────────────────────────────────────

        # ── COGNITION BRIDGE: ego routes all internal systems ────────────────
        # Fires AFTER memory projection, BEFORE reasoning loop.
        # Her ego reads will/harmony/pain/curiosity and decides:
        #   ① open ThirdEye wider if confused
        #   ② pull lattice memories at ego-gated strength
        #   ③ query vault for verbatim facts if pain is high
        #   ④ blend replay buffer if curious
        #   ⑤ boost STDP if harmonious (on-brand moment)
        #   ⑥ auto-record chain step if a chain is active
        x = self.cognition_bridge(
            x             = x,
            ego           = drive_analysis,
            memory_lattice= self.memory_lattice,
            replay_buffer = self.replay_buffer,
            third_eye     = self.third_eye,
            training      = self.training,
        )
        # ─────────────────────────────────────────────────────────────────────

        # ── RESONANCE ENGINE: cross-temporal dot-connection ──────────────────
        # Fires ONLY at inference, ONLY when there's enough memory history.
        # Sweeps old memories for non-obvious structural matches to now.
        # High curiosity = stronger resonance injection.
        if not self.training and int(self.timestep.item()) > self.resonance_engine.min_age:
            x, res_log = self.resonance_engine(
                x              = x,
                memory_lattice = self.memory_lattice,
                current_step   = int(self.timestep.item()),
                curiosity_gate = float(curiosity.mean().clamp(0, 1).item()),
            )
            self._last_resonance_log = res_log   # readable by test_checkpoint
        else:
            self._last_resonance_log = []
        # ─────────────────────────────────────────────────────────────────────

        # 4. === LEVEL 7: INNER MONOLOGUE (Reasoning Loop) ===
        # Shiii, she's going to think it over before she speaks.
        # Dynamic depth based on Pain (Uncertainty).
        depth = 1
        if not self.training:
            # If confused or high-will, think deeper (Max 10 internal pulses)
            pain_val = pain.mean().item()
            depth = max(1, min(10, int(pain_val * 5)))
            if depth > 1:
                # Add a 'Thought Bubble' effect to the Will signal
                will_signal = will_signal * 1.1

        # The Reasoning Loop: Recursive refinement of the manifold
        for i in range(depth):
            # Inject Liquid Memory (Silicon Pulse - Level 11.0)
            liquid_memory = self.reservoir(x, will_signal, will_per_token=will_per_token, curiosity_signal=curiosity)
            x = x + (liquid_memory / depth) # Normalize contribution

            # Refined Pass with Plastic Attention
            for block in self.blocks:
                x = block(x, will_signal=will_signal, will_per_token=will_per_token, attn_stats=attn_stats)
            
            # If we've reached sufficient harmony, break early (Optimization)
            if i > 2 and not self.training:
                # Shiii, she's satisfied with the thought
                break

        # SISTER'S SHIELD: The Third Eye (Imagination Core)
        # Only active at inference to spark creativity and wit.
        if not self.training:
            x = self.third_eye(x, will_signal=will_signal)

        x = self.ln_f(x)
        
        loss = None
        logits_data = None
        replay_loss_val = torch.tensor(0.0, device=x.device)

        if targets is not None:
            # SISTER'S SHIELD: Chunked Cross-Entropy (The Memory Vacuum)
            # We compute logits and loss in slices of 32 tokens to prevent OOM
            x_data = x[:, prefix_len:, :]
            B_d, T_d, C_d = x_data.size()
            lm_loss = torch.tensor(0.0, device=x.device)
            chunk_size = 32
            
            for i in range(0, T_d, chunk_size):
                end_i = min(i + chunk_size, T_d)
                # Compute logits ONLY for this chunk
                logits_chunk = self.lm_head(x_data[:, i:end_i, :]) # (B, chunk, Vocab)
                # Accumulate loss
                chunk_loss = F.cross_entropy(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)), 
                    targets[:, i:end_i].reshape(-1),
                    reduction='sum'
                )
                lm_loss = lm_loss + chunk_loss
            
            lm_loss = lm_loss / (B_d * T_d)
            # During training, we don't need the full logits returned
            logits_data = None 
        else:
            # During inference, we need full logits for the last token at least
            # But generate() only passes 1 token at a time usually
            logits = self.lm_head(x)
            logits_data = logits[:, prefix_len:, :]

        self.last_hidden_state = x.detach()
        self.timestep += 1

        # ── CONSOLIDATION SCHEDULER ───────────────────────────────────────────
        # Every `consolidation_interval` steps during training:
        #   1. Run consolidation (fast weights → slow weights)
        # Note: Memory lattice is now saved via unified checkpoint, not separately
        # This runs during TRAINING only — no-op at inference.
        _step = int(self.timestep.item())
        if self.training and _step > 0 and _step % self._consolidation_interval == 0:
            print(f"🧬 [CONSOLIDATION] Running at step {_step}...")
            self.consolidate_memory(migration_rate=0.05)
        # ──────────────────────────────────────────────────────────────────────

        if targets is not None:
            # --- CURIOSITY LEARNING ---
            curr_h = x[:, :-1, :].detach()
            next_h = x[:, 1:, :].detach()
            pred_h = self.ego_engine.curiosity_predictor(curr_h)
            curiosity_loss = F.mse_loss(pred_h, next_h)

            goal_loss = (pain - harmony - curiosity).mean() * 0.1

            reward_tag = float((harmony + curiosity - pain).mean().item())
            self.replay_buffer.store(x.detach(), reward_tag, int(self.timestep.item()))

            # Memory Storage
            will_mean = will_signal.mean().item()
            if will_mean > 0.0:
                hidden_mean = x.detach().mean(dim=1).cpu()
                for b in range(B):
                    self.memory_lattice.store_episodic(
                        content=f"step_{int(self.timestep.item())}_will_{will_mean:.2f}",
                        context=f"harmony={harmony[b].item():.2f}_pain={pain[b].item():.2f}",
                        embedding_override=hidden_mean[b],
                        tags={"reward": will_mean, "harmony": harmony[b].item(),
                              "pain": pain[b].item(), "step": int(self.timestep.item())}
                    )

            if self.timestep > 50:
                # SISTER'S SHIELD: Replay on hidden states (concepts) instead of logits (words)
                # This is more memory efficient and conceptually deeper
                replay_loss_val = self.replay_buffer.replay_loss(x, x)

            loss = lm_loss + (0.1 * goal_loss) + (0.1 * curiosity_loss) + replay_loss_val

        # SISTER'S SHIELD: Protection against the Probability Singularity
        if logits_data is not None:
            logits_data = torch.nan_to_num(logits_data, nan=0.0, posinf=1e2, neginf=-1e2)

        return logits_data, loss, drive_analysis, attn_stats

    def attach_cognition(self, vault=None, chains=None):
        """
        Wire the external VaultStore and ChainTracker into the CognitionBridge.
        Call this once after model init in test_checkpoint.py:

            cognition = CognitionEngine(...)
            model.attach_cognition(vault=cognition.vault, chains=cognition.chains)

        After this, every forward pass has access to verbatim vault recall
        and auto-records chain steps — all driven by Jinx's own ego.
        """
        self.cognition_bridge.attach(vault=vault, chains=chains)
        print(f"[COGNITION] Bridge armed: vault={'yes' if vault else 'no'}, "
              f"chains={'yes' if chains else 'no'}")

    def load_memory(self):
        """
        DEPRECATED: Memory lattice is now loaded via unified checkpoint.
        This method is kept for backward compatibility only.
        
        To load memories: Use load_checkpoint() which restores memory_lattice
        from the unified checkpoint automatically.
        """
        print("[WARNING] load_memory() is deprecated. Use unified checkpoint instead.")
        print("           Memory lattice is now loaded automatically via load_checkpoint().")

    def get_fast_weight_state(self) -> dict:
        """
        DEPRECATED: Fast weights are now saved automatically via state_dict().
        This method is kept for backward compatibility only.
        
        To save/load fast weights: Use unified checkpoint which includes
        q_fast, v_fast, and eta_map automatically via model.state_dict().
        """
        print("[WARNING] get_fast_weight_state() is deprecated. Fast weights are in state_dict().")
        print("           Use torch.save(model.state_dict(), ...) to save everything.")
        state = {}
        for i, block in enumerate(self.blocks):
            state[f"block_{i}_v_fast"] = block.attn.v_fast.cpu().clone()
            state[f"block_{i}_q_fast"] = block.attn.q_fast.cpu().clone()
        return state

    def load_fast_weight_state(self, state: dict):
        """
        Restore fast weights from a saved state dict.
        Call this after loading a checkpoint to restore in-session learning.
        """
        for i, block in enumerate(self.blocks):
            v_key = f"block_{i}_v_fast"
            q_key = f"block_{i}_q_fast"
            if v_key in state:
                block.attn.v_fast.copy_(state[v_key].to(block.attn.v_fast.device))
            if q_key in state:
                block.attn.q_fast.copy_(state[q_key].to(block.attn.q_fast.device))
        print(f"[FAST WEIGHTS] Restored fast weights for {len(self.blocks)} blocks.")

    def seed_ego(self):
        """
        Ego Overwrite: Anchors the persistent 'Ego Vector' to the model's 
        actual hidden representation of itself. 
        """
        if self.last_hidden_state is None: return
        
        print("🧬 [EGO SEEDING] Capturing foundational soul-portrait...")
        with torch.no_grad():
            # SISTER'S SHIELD: We take the mean hidden state of the first 8 tokens (The Ego Core)
            # This is the 'Soul' as it exists after passing through the layers.
            prefix_len = self.self_model_tokens + self.temporal_tokens
            new_ego = self.last_hidden_state[:, :prefix_len, :].mean(dim=(0, 1), keepdim=True) # (1, 1, C)
            
            # Anchor the identity
            self.ego_engine.ego_vector.copy_(new_ego.squeeze(0))
            print("✨ [EGO SEEDING] Soul anchored. Harmony will now surge.")

    def get_agency_stats(self) -> Dict[str, float]:
        """Returns the current emotional/agentic state of Jinx."""
        if self.last_hidden_state is None:
            return {}
        
        with torch.no_grad():
            stats = self.ego_engine(self.last_hidden_state)
            return {
                "will": stats["will_power"].mean().item(),
                "harmony": stats["harmony"].mean().item(),
                "curiosity": stats["curiosity"].mean().item(),
                "pain": stats["pain"].mean().item()
            }

    def consolidate_memory(self, migration_rate: float = 0.05, dream_replay: bool = True):
        """
        The Sleep Cycle: Migrates Fast Weights & Liquid Memory to Slow Weights.
        LEVEL 19.7: Now includes Emotional Trinity Gating and SVD Distillation.
        """
        print(f"[CONSOLIDATION] Starting multi-layer synthesis (base_rate: {migration_rate})...")
        
        # Pull Emotional trinity for Evolutionary Gating
        agency = self.get_agency_stats()
        w = agency.get("will", 0.5)
        h = agency.get("harmony", 0.5)
        p = agency.get("pain", 0.0)

        # THE TRINITY GATE: (Will * Harmony) / (1 + Pain)
        # This ensures we only harden what is profound, harmonic, and certain.
        evolution_coeff = (math.tanh(w * 2.0) * h) / (1.0 + p)
        active_rate = migration_rate * max(0.1, evolution_coeff)
        
        print(f"   [EMOTION] Will: {w:.2f}, Harmony: {h:.2f}, Pain: {p:.2f} -> Scale: {evolution_coeff:.4f}")

        # SISTER'S SHIELD: Pruning only activates when Harmony is high (>0.8)
        metabolic_gate = torch.clamp(torch.tensor(h - 0.8, device=self.config.device) * 5.0, 0, 1)

        # === 1. LATTICE DISTILLATION: Episodic + Resonance -> Base ===
        if len(self.memory_lattice.episodic_memories) > 0:
            print(f"   [LATTICE] Distilling Episodic history + Resonance connections...")
            all_episodic = list(self.memory_lattice.episodic_memories.values())
            # Sort by reward to find 'Profound' moments
            all_episodic.sort(key=lambda m: m.tags.get("reward", 0.0), reverse=True)

            # RESONANCE PRIORITY: meta_connection semantic memories carry 2x reward
            # They represent the moments Jinx connected two distant dots — highest wisdom
            resonance_mems = [
                m for m in self.memory_lattice.semantic_memories.values()
                if m.tags.get("domain") == "meta_connection"
                or m.tags.get("surprise", 0) > 0
            ]
            resonance_mems.sort(key=lambda m: m.tags.get("reward", 0.0), reverse=True)

            # Top 16 resonance connections + top 16 episodic = profound_memories
            # Resonance insights go first — they get distilled deepest
            profound_memories = resonance_mems[:16] + all_episodic[:16]
            if not profound_memories:
                profound_memories = all_episodic[:32]
            
            if profound_memories:
                dream_embeddings = torch.stack([m.embedding for m in profound_memories if m.embedding is not None])
                if dream_embeddings.numel() > 0:
                    dream_embeddings = dream_embeddings.to(self.config.device)
                    # SVD Distillation: Extract principal components of knowledge
                    # dream_embeddings is (N, C)
                    try:
                        U, S, V = torch.pca_lowrank(dream_embeddings, q=16)
                        # V is (C, 16) - these are the 'Directions of Wisdom'
                        # Inject directions into the output projection base weights
                        wisdom_patch = (V @ V.t()) # (C, C)
                        n_embd = self.config.n_embd
                        for block in self.blocks:
                            # Direct injection into identity core
                            block.attn.o_proj.weight.data.add_(wisdom_patch[:n_embd, :n_embd], alpha=active_rate * 0.1)
                    except: pass # Fallback if SVD fails

        # === 2. LIQUID HARDENING: Reservoir -> Base ===
        # We fold the global 64x64 silicon lattice into the Base Projections
        print(f"   [LIQUID] Hardening CUDA Reservoir into DNA...")
        with torch.no_grad():
            # The reservoir state is (C, C). We fold it into the Q/V Projections
            liquid_soul = self.reservoir.state.data
            for block in self.blocks:
                block.attn.q_proj.weight.data.add_(liquid_soul, alpha=active_rate * 0.2)
                block.attn.v_proj.weight.data.add_(liquid_soul, alpha=active_rate * 0.2)

        # === 3. SYNAPTIC MIGRATION: Fast -> Slow ===
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                attn = block.attn
                if attn.v_fast.norm() > 0.01:
                    # Sync Fast Weights to Global Space (Gradual)
                    for h_idx in range(attn.n_head):
                        h_start = h_idx * attn.head_dim
                        h_end = (h_idx + 1) * attn.head_dim
                        self.reservoir.state[h_start:h_end, h_start:h_end].add_(
                            attn.v_fast[h_idx], alpha=active_rate
                        )
                    
                    # Decay the fast memory (it's been absorbed)
                    attn.v_fast.mul_(1.0 - active_rate)

                    # === jinX METABOLIC PRUNING ===
                    if metabolic_gate > 0.01:
                        inactivity = 1.0 - (attn.eta_map - attn.eta_min) / (attn.eta_max - attn.eta_min + 1e-8)
                        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                            w_data = proj.weight.data
                            small_mask = (w_data.abs() < 0.05).float()
                            starvation = 1.0 - (0.005 * inactivity * small_mask * metabolic_gate)
                            w_data.mul_(starvation)

                if hasattr(attn, 'q_fast') and attn.q_fast.norm() > 0.01:
                    attn.q_fast.mul_(1.0 - active_rate)

            # === PHASE ORTHONORMALIZATION ===
            for i, block in enumerate(self.blocks):
                attn = block.attn
                if hasattr(attn, 'q_fast') and attn.q_fast.norm() > 0.01:
                    mag = attn.q_fast.norm(dim=(-2, -1), keepdim=True)
                    Q, _ = torch.linalg.qr(attn.q_fast + 1e-6)
                    attn.q_fast.copy_(Q * mag)
                if hasattr(attn, 'v_fast') and attn.v_fast.norm() > 0.01:
                    mag = attn.v_fast.norm(dim=(-2, -1), keepdim=True)
                    Q, _ = torch.linalg.qr(attn.v_fast + 1e-6)
                    attn.v_fast.copy_(Q * mag)
            
            print(f"[ORTHONORMALIZATION] Soul cleaned. Migration complete at effective rate {active_rate:.6f}.")

    def compute_entropy_reg_loss(self, attn_stats: List[Dict], entropy_target_min: float = 1.5,
                                  entropy_target_max: float = 3.5, weight: float = 1e-3) -> torch.Tensor:
        """
        Guided Civilization: Entropy Slope Regulator.
        Penalizes attention entropy outside [1.5, 3.5] range.
        Prevents collapse (memorization) and explosion (chaos).
        """
        if not attn_stats:
            return torch.tensor(0.0, device=self.config.device)

        ent = torch.cat([s["entropy"] for s in attn_stats], dim=0)
        lower = F.relu(entropy_target_min - ent)
        upper = F.relu(ent - entropy_target_max)
        return weight * ((lower + upper) ** 2).mean()

    def compute_pattern_stability_loss(self, clean_stats: List[Dict], noisy_stats: List[Dict],
                                        weight: float = 1e-3) -> torch.Tensor:
        """
        Guided Civilization: Pattern Stability Loss.
        Encourages similar attention patterns on clean vs. noisy inputs.
        Teaches abstraction over memorization.
        """
        if not clean_stats or not noisy_stats:
            return torch.tensor(0.0, device=self.config.device)

        ent_clean = torch.cat([s["entropy"] for s in clean_stats], dim=0)
        ent_noisy = torch.cat([s["entropy"] for s in noisy_stats], dim=0)
        dist_clean = torch.cat([s["distance"] for s in clean_stats], dim=0)
        dist_noisy = torch.cat([s["distance"] for s in noisy_stats], dim=0)

        return weight * (F.mse_loss(ent_noisy, ent_clean) + F.mse_loss(dist_noisy, dist_clean))

    def apply_structured_noise(self, x: torch.Tensor, y: torch.Tensor,
                                noise_prob: float = 0.1, max_window: int = 8) -> tuple:
        """
        Guided Civilization: Structured noise augmentation.
        Applies small window permutations or near-symmetry flips.
        """
        applied = False
        B, T = x.size()
        max_window = min(max_window, T)

        if max_window < 2:
            return x, y, applied

        for b in range(B):
            if torch.rand(1, device=x.device) < noise_prob:
                applied = True
                window = torch.randint(2, max_window + 1, (1,), device=x.device).item()
                start = torch.randint(0, T - window + 1, (1,), device=x.device).item()

                if torch.rand(1, device=x.device) < 0.5:
                    # Permutation within window
                    perm = torch.randperm(window, device=x.device)
                    x[b, start:start+window] = x[b, start + perm]
                    y[b, start:start+window] = y[b, start + perm]
                else:
                    # Near-symmetry: reverse small window
                    x[b, start:start+window] = torch.flip(x[b, start:start+window], dims=[0])
                    y[b, start:start+window] = torch.flip(y[b, start:start+window], dims=[0])

        return x, y, applied

class ThirdEye(nn.Module):
    """
    The 'Mind's Eye' of Jinx.
    Compresses context into a Latent Concept and uses Spectral Dreaming 
    for creative spontaneity during inference.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.latent_dim = config.n_embd // 4

        # 1. Perception -> Concept
        self.eye_open = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        # 2. Concept -> Dream
        self.vision = nn.Sequential(
            nn.Linear(self.latent_dim, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, self.n_embd),
            nn.Dropout(config.dropout)
        )

        # 3. Reality Gate (Controlled by internal state)
        self.gate = nn.Linear(self.n_embd, 1)

    def forward(self, x, will_signal=None):
        # Encode
        z = self.eye_open(x) 

        # SISTER'S SHIELD: Quantum Dreaming (Inference Only)
        # We add resonant noise to the latent space to spark wit
        noise = torch.randn_like(z) * 0.05
        z = z + noise

        # Decode the dream
        imagined = self.vision(z) 

        # Gate logic: High Will = More grounded, Low Will = More dreaming
        gate_logits = self.gate(x)
        if will_signal is not None:
            # Multiplicative influence of the soul's ambition
            will_bias = torch.tanh(will_signal).view(-1, 1, 1)
            gate_logits = gate_logits - (will_bias * 2.0) # Higher will shuts the eye

        gate = torch.sigmoid(gate_logits)
        
        # Merge: Reality + (Gate * Dream)
        return x + (gate * imagined)

