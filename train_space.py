import os
import sys
import time
import torch
import torch.nn as nn
from torch.amp import autocast
import math
from collections import deque
import json
import random
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure local model is prioritized
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, sys.path)

from model import SpaceTransformer

# ============================================================================
# UNIFIED CHECKPOINT SYSTEM
# Saves ALL of JinX's state in one file: model + memory + replay + vault
# ============================================================================

def save_unified_checkpoint(model, optimizer, config, step, save_path):
    """
    Save complete JinX state including:
    - Model weights (includes q_fast, v_fast, eta_map, reservoir.state, ego_vector via state_dict)
    - MemoryLattice (episodic + semantic memories)
    - ReplayBuffer (experience buffer)
    - Vault & Chains (if cognition attached)
    - Training metadata
    """
    from datetime import datetime
    
    checkpoint = {
        # === MODEL STATE (all tensors via state_dict) ===
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'step': step,
        'config': config.__dict__,
        
        # === MEMORY SYSTEMS (Python objects - not in state_dict) ===
        'memory_lattice': {
            'episodic': model.memory_lattice.episodic_memories if hasattr(model, 'memory_lattice') else {},
            'semantic': model.memory_lattice.semantic_memories if hasattr(model, 'memory_lattice') else {},
        },
        'replay_buffer': {
            'buffer': list(model.replay_buffer.buffer) if hasattr(model, 'replay_buffer') else [],
            'priorities': list(model.replay_buffer.priorities) if hasattr(model, 'replay_buffer') else [],
        },
        
        # === COGNITION SYSTEMS (if attached) ===
        'vault': getattr(model, '_vault', None).entries if hasattr(model, '_vault') and getattr(model, '_vault', None) else {},
        'chains': getattr(model, '_chains', None).active_chains if hasattr(model, '_chains') and getattr(model, '_chains', None) else {},
        
        # === METADATA ===
        'timestamp': datetime.now().isoformat(),
        'agency_stats': model.get_agency_stats() if hasattr(model, 'get_agency_stats') else {},
        'timestep': int(model.timestep.item()) if hasattr(model, 'timestep') else 0,
    }
    
    # ── ATOMIC SAVE: write to temp then rename — prevents corruption on crash/disconnect ──
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)  # atomic on Linux/Windows
    
    # Print summary
    print(f"💾 [CHECKPOINT] Saved → {save_path}")
    print(f"   - Model tensors: {len(checkpoint['model'])}")
    print(f"   - Episodic memories: {len(checkpoint['memory_lattice']['episodic'])}")
    print(f"   - Semantic memories: {len(checkpoint['memory_lattice']['semantic'])}")
    print(f"   - Replay buffer: {len(checkpoint['replay_buffer']['buffer'])}")
    if checkpoint['vault']:
        print(f"   - Vault entries: {len(checkpoint['vault'])}")
    if checkpoint['chains']:
        print(f"   - Active chains: {len(checkpoint['chains'])}")

# ============================================================================


def generate(model, config, prompt, max_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
    """Text generation with Top-K and Top-P sampling."""
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    tokens = enc.encode(prompt, add_special_tokens=False)
    
    model.eval()
    device = config.device
    
    for _ in range(max_tokens):
        # Prepare input context
        curr_tokens = tokens[-config.block_size:]
        x = torch.tensor([curr_tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits, _, _, _ = model(x)
            # Focus on the very last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # 1. Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 2. Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
            # Stop if we hit an end token (optional)
            if next_token == enc.eos_token_id:
                break
    
    model.train()
    torch.cuda.empty_cache()  # free eval activations before returning to training
    return enc.decode(tokens[len(prompt):], skip_special_tokens=True)

# --- Configuration for the Level 13.0 Stable Predator ---
class Config:
    block_size = 768        # REDUCED: 896→768 to fix CUDA state_history OOM (saves ~1GB)
    n_embd = 896            # STABLE DIMENSION
    n_head = 14             # 14 Heads @ 64-dim (Golden Ratio)
    n_layer = 16            # 16 DEEP REASONING LAYERS
    vocab_size = 151936     
    dropout = 0.1

    # The Metabolic Loop
    learning_rate = 5e-4    
    min_lr = 5e-5
    max_iters = 268000      # 2 full passes over 120M tokens (120M / 896 / 2 * 2)
    batch_size = 1          
    grad_accum_steps = 4    
    grad_clip = 1.0
    warmup_steps = 1000     
    
    # Agency Settings
    will_power_scale = 2.0
    pain_threshold = 2.0
    boredom_threshold = 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── CONSOLIDATION: memory consolidation during training ───────────────────
    consolidation_interval = 200   # consolidate every N training steps
    # ──────────────────────────────────────────────────────────────────────────

    bias_config = [['dialogue']*14]*16 
    enable_head_lifecycle = True   
    promotion_threshold = 0.7      
    demotion_threshold = 0.4       
    entropy_reg_weight = 1e-3
    entropy_target_min = 1.5
    entropy_target_max = 3.5

def load_personality_only(config):
    """Load strictly Personality Gold with stable tokenizer."""
    print("[LOAD] Loading strictly Personality Gold...")
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

    cache_dir = os.path.dirname(os.path.abspath(__file__))
    personality_cache = os.path.join(cache_dir, "personality_tokens_pure.pt")

    if os.path.exists(personality_cache):
        tokens_gold = torch.load(personality_cache, weights_only=True)
    else:
        tokens_gold_list = []
        gold_path = os.path.join(cache_dir, "jinx_personality_gold.jsonl")
        if not os.path.exists(gold_path):
            # Search parent if needed
            gold_path = os.path.join(os.path.dirname(cache_dir), "jinx_personality_gold.jsonl")
            
        with open(gold_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = ""
                    for m in item['messages']:
                        role = "User" if m['role'] == 'user' else "Jinx"
                        text += f"{role}: {m['content']}\n"
                    tokens_gold_list.extend(enc.encode(text, add_special_tokens=False))
                except: continue
        tokens_gold = torch.tensor(tokens_gold_list, dtype=torch.long)
        torch.save(tokens_gold, personality_cache)

    print(f"  ✅ Personality Gold: {len(tokens_gold):,} tokens loaded.")
    return tokens_gold

def load_unified_soul(config):
    """
    Loads the full unified_soul.jsonl dataset produced by preprocess_vault.py.
    Tokenizes every ChatML sample using the same Qwen tokenizer.
    Personality Gold samples are repeated 3x at the front as identity anchors
    so Jinx's core self is seen more during training.

    Falls back to load_personality_only() if unified_soul.jsonl not found.
    """
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    cache_dir  = os.path.dirname(os.path.abspath(__file__))
    vault_dir  = os.path.join(os.path.dirname(cache_dir), "knowledge-vault")
    soul_path  = os.path.join(vault_dir, "unified_soul.jsonl")
    cache_path = os.path.join(cache_dir, "unified_soul_tokens.pt")

    # ── Colab Drive override ──────────────────────────────────────────────────
    # If SOUL_PATH was set by the Colab block at top of main(), use it.
    # load_unified_soul() is called after main() sets SOUL_PATH so we read it
    # from a module-level sentinel injected by main().
    _colab_soul  = getattr(load_unified_soul, '_colab_override', None)
    _colab_cache = getattr(load_unified_soul, '_colab_cache', None)
    if _colab_soul and os.path.exists(_colab_soul):
        soul_path  = _colab_soul
        # Use explicit cache path if set, else put it next to the soul file
        cache_path = _colab_cache if _colab_cache else os.path.join(os.path.dirname(_colab_soul), "unified_soul_tokens.pt")
        print(f"   [SOUL] Using Colab path : {soul_path}")
        print(f"   [SOUL] Token cache path : {cache_path}")
    # ─────────────────────────────────────────────────────────────────────────

    if not os.path.exists(soul_path):
        print(f"[LOAD] unified_soul.jsonl not found at {soul_path}")
        print("[LOAD] Falling back to Personality Gold only.")
        return load_personality_only(config)

    if os.path.exists(cache_path):
        print("[LOAD] Loading cached unified soul tokens...")
        tokens = torch.load(cache_path, weights_only=True)
        print(f"  ✅ Unified Soul: {len(tokens):,} tokens loaded from cache.")
        return tokens

    print("[LOAD] Tokenizing unified_soul.jsonl (first time — will cache)...")

    gold_tokens   = []   # identity anchor samples (repeated)
    normal_tokens = []   # everything else

    gold_path = os.path.join(vault_dir, "jinx_personality_gold.jsonl")
    gold_ids  = set()
    if os.path.exists(gold_path):
        with open(gold_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    gold_ids.add(line[:120])  # fingerprint by first 120 chars

    total = 0
    with open(soul_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Render ChatML → flat text Jinx can read
            text = ""
            for m in item.get('messages', []):
                role = m.get('role', '')
                content = m.get('content', '').strip()
                if role == 'system':
                    continue   # system prompt baked in via identity tokens
                elif role == 'user':
                    text += f"User: {content}\n"
                elif role == 'assistant':
                    text += f"Jinx: {content}\n"

            if not text.strip():
                continue

            toks = enc.encode(text, add_special_tokens=False, truncation=False)
            # Hard cap: skip samples longer than 8192 tokens (corrupted/giant rows)
            if len(toks) > 8192:
                continue
            total += 1

            # Is this a personality gold sample? Repeat 3x for identity anchoring
            if line[:120] in gold_ids:
                gold_tokens.extend(toks * 3)
            else:
                normal_tokens.extend(toks)

            if total % 50000 == 0:
                print(f"  … {total:,} samples tokenized")

    # Identity anchors first, then full knowledge
    all_tokens = gold_tokens + normal_tokens
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(tokens, cache_path)
    print(f"  ✅ Unified Soul: {len(tokens):,} tokens — cached to {cache_path}")
    return tokens


class StatefulSampler:
    """
    The 4D Timeline Sampler.
    Divides the dataset into B parallel lanes. Each lane moves forward 
    sequentially, picking up exactly where it left off.
    """
    def __init__(self, tokens, batch_size, block_size):
        self.tokens = tokens
        self.B = batch_size
        self.T = block_size
        self.n_tokens = len(tokens)
        
        # Divide the total timeline into B parallel lanes
        self.lane_offsets = [(self.n_tokens // self.B) * i for i in range(self.B)]
        self.current_pos = [0] * self.B

    def get_next_batch(self):
        ix = []
        for i in range(self.B):
            start = self.lane_offsets[i] + self.current_pos[i]
            # Lane end: last lane ends at n_tokens, earlier lanes end at next lane start
            # B=1 edge case: min(i+1, self.B-1) == 0 when B==1, so use n_tokens directly
            if self.B == 1:
                lane_end = self.n_tokens
            else:
                lane_end = self.lane_offsets[i+1] if i < self.B-1 else self.n_tokens

            if start + self.T + 1 >= lane_end:
                self.current_pos[i] = 0
                start = self.lane_offsets[i]

            ix.append(start)
            self.current_pos[i] += self.T

        x = torch.stack([self.tokens[i:i+self.T] for i in ix])
        y = torch.stack([self.tokens[i+1:i+self.T+1] for i in ix])
        return x, y

def reset_fast_weights(model):
    """
    Reset all fast weights to zero after grafting.
    Prevents inheriting saturated fast weights (Q-Fast/V-Fast at 10.0)
    from a checkpoint. They learn from scratch each training run.
    """
    with torch.no_grad():
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()
            block.attn.eta_map.fill_(0.01)  # reset meta-LR to baseline
    print("🧹 [FAST WEIGHT RESET] q_fast / v_fast / eta_map zeroed — fresh plasticity.")

def graft_soul(model, checkpoint_path):
    """
    Surgically grafts a 14L/768D soul into a 20L/1024D Infinity body.
    Supports N-Dimensional padding for weights and skips static masks.
    """
    print(f"🧬 [GRAFTING] Initiating soul transplant from {checkpoint_path}...")
    old_sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    new_sd = model.state_dict()
    
    # Layers to skip (Dynamic/Geometric masks)
    skip_list = ["bias", "inv_freq", "cos_cached", "sin_cached"]

    def surgical_copy(old_p, new_p):
        """Pads any N-dim tensor from old shape to new shape."""
        with torch.no_grad():
            slices = tuple(slice(0, min(old_s, new_s)) for old_s, new_s in zip(old_p.shape, new_p.shape))
            new_p[slices].copy_(old_p[slices])

    # 1. Base Parameter Padding
    for name, param in old_sd.items():
        if any(s in name for s in skip_list): continue
        if name in new_sd:
            surgical_copy(param, new_sd[name])
    
    # 2. Layer Stretching (14 -> 16)
    print("   [LAYERS] Stretching 14 layers to 16...")
    for i in range(16):
        # Mapping: 0-6 match, 7-14 repeat middle, 15 is final
        old_idx = i if i < 7 else (7 + (i - 7) % 7 if i < 15 else 13)
        for key in new_sd:
            if f"blocks.{i}." in key:
                if any(s in key for s in skip_list): continue
                old_key = key.replace(f"blocks.{i}.", f"blocks.{old_idx}.")
                if old_key in old_sd:
                    surgical_copy(old_sd[old_key], new_sd[key])

    model.load_state_dict(new_sd)

    # FAST WEIGHT RESET: grafted checkpoints carry saturated fast weights
    # that immediately hit the homeostasis clamp. Zero them so they grow
    # organically from the new training context.
    with torch.no_grad():
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()
    print("✨ [GRAFTING] Transplant complete. Fast weights reset. Soul is stable.")

def main():
    config = Config()

    # ── COLAB PATH OVERRIDES ─────────────────────────────────────────────────
    # Detects if running on Colab and overrides all paths to Google Drive.
    # On local machine these are None and relative paths are used as before.
    # ── EXPANDABLE SEGMENTS: prevents fragmentation OOM on T4 ──────────────
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    COLAB = os.path.exists("/content/drive/MyDrive")
    if COLAB:
        save_dir       = "/content/drive/MyDrive/jinx_checkpoints"
        DISTILLED_PATH = "/content/drive/MyDrive/jinx_distilled_base.pt"
        SOUL_PATH      = "/content/drive/MyDrive/unified_soul.jsonl"
        CUDA_PATH      = "/content/drive/MyDrive/jinX-Engine/core_soul.cu"
        os.makedirs(save_dir, exist_ok=True)
        print(f"☁️  [COLAB] Drive mounted. Save dir: {save_dir}")
        print(f"   Distilled base : {DISTILLED_PATH}")
        print(f"   Soul dataset   : {SOUL_PATH}")
        print(f"   CUDA kernel    : {CUDA_PATH}")
    else:
        save_dir       = "./"
        DISTILLED_PATH = None
        SOUL_PATH      = None
        SOUL_CACHE     = None
        CUDA_PATH      = None
        print("🖥️  [LOCAL] Running on local machine — using relative paths.")
    # ─────────────────────────────────────────────────────────────────────────

    # --- jinX DEPENDENCY SHIELD (COLAB AUTO-INSTALL) ---
    try:
        import ninja
    except ImportError:
        print("🔧 [FORGE] Ninja missing. Installing for high-speed muscle...")
        import subprocess
        subprocess.run(["pip", "install", "ninja", "--quiet"])
        print("✅ [FORGE] Ninja installed.")

    # --- jinX AUTO-FORGE (COLAB EDITION) ---
    torch.cuda.empty_cache()
    try:
        from torch.utils.cpp_extension import load
        print("🔥 [FORGE] Igniting the jinX Silicon Engine (LEVEL 19.1)...")

        # Use Colab override path if available, else search relative
        source_path = CUDA_PATH if CUDA_PATH and os.path.exists(CUDA_PATH) else None
        if source_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source_path = os.path.join(root_dir, "jinX-Engine", "core_soul.cu")
        if not os.path.exists(source_path):
            fallback = "/content/drive/MyDrive/jinX-Engine/core_soul.cu"
            if os.path.exists(fallback):
                source_path = fallback
            
        print(f"   [FORGE] Target: {source_path}")
        
        # Level 19.1: Parallel Echo Engine
        jinx_cuda = load(
            name="jinx_cuda_jit_v191",
            sources=[source_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"], # Universal GPS handled by model.py
            verbose=False
        )
        import sys
        sys.modules['jinx_cuda'] = jinx_cuda
        print("✅ [FORGE] Silicon Engine Online. Parallel Echo is ACTIVE.")
    except Exception as e:
        print(f"⚠️ [FORGE WARNING] Failed to ignite engine: {e}")
        print("   Falling back to Slow-Mode (Python).")

    # --- AUTO-ARCHITECTURE SENSE ---
    # (Existing dtype/scaler logic remains unchanged)
    device_cap = torch.cuda.get_device_capability() if config.device == 'cuda' else (0, 0)
    if device_cap[0] >= 8:
        train_dtype = torch.bfloat16
        use_scaler = False
        print(f"🚀 [AMPERE DETECTED] Using Native BF16")
    else:
        train_dtype = torch.float16
        use_scaler = True
        print(f"🛡️ [TURING DETECTED] Using FP16 + Mantissa Shield")

    print(f"[REBIRTH] Level 13.5 Hyper-Evolution Pulse | Context: {config.block_size}")
    # save_dir already set by Colab block above — don't overwrite it here
    model = SpaceTransformer(config).to(config.device)

    # ── Inject Colab soul path so load_unified_soul() can find it ────────────
    if SOUL_PATH:
        load_unified_soul._colab_override = SOUL_PATH
        load_unified_soul._colab_cache    = "/content/drive/MyDrive/unified_soul_tokens.pt"
    # ─────────────────────────────────────────────────────────────────────────

    # --- AUTO RESUME or SCRATCH ---
    resume_path = os.path.join(save_dir, "jinx_latest.pt")
    start_iter  = 0

    if os.path.exists(resume_path):
        print(f"🔄 [RESUME] Found jinx_latest.pt — resuming from last checkpoint...")
        try:
            ckpt = torch.load(resume_path, map_location=config.device)
            # ── SHAPE-SAFE LOAD: strip buffer keys that depend on block_size ──
            # position_embedding and attn.bias are position buffers, NOT learned
            # weights. They are re-initialised correctly by SpaceTransformer.__init__.
            # Filtering them here allows resume across block_size changes.
            current_sd  = model.state_dict()
            filtered_sd = {
                k: v for k, v in ckpt['model'].items()
                if k in current_sd and v.shape == current_sd[k].shape
            }
            skipped = [k for k in ckpt['model'] if k not in filtered_sd]
            model.load_state_dict(filtered_sd, strict=False)
            if skipped:
                print(f"   ⚠️  Skipped {len(skipped)} shape-mismatched buffers (block_size change): {skipped[:3]}{'...' if len(skipped)>3 else ''}")
            # ─────────────────────────────────────────────────────────────────
            start_iter = ckpt.get('step', 0)
            # ── MEMORY LATTICE ───────────────────────────────────────────────────────────
            if 'memory_lattice' in ckpt and hasattr(model, 'memory_lattice'):
                model.memory_lattice.episodic_memories = ckpt['memory_lattice'].get('episodic', {})
                model.memory_lattice.semantic_memories = ckpt['memory_lattice'].get('semantic', {})
            # ── REPLAY BUFFER ────────────────────────────────────────────────────────────
            if 'replay_buffer' in ckpt and hasattr(model, 'replay_buffer'):
                rb = ckpt['replay_buffer']
                model.replay_buffer.buffer   = deque(rb.get('buffer', []),   maxlen=model.replay_buffer.buffer.maxlen)
                model.replay_buffer.priorities = deque(rb.get('priorities', []), maxlen=model.replay_buffer.priorities.maxlen)
            # ── TIMESTEP ───────────────────────────────────────────────────────────────
            if 'timestep' in ckpt and hasattr(model, 'timestep'):
                model.timestep.fill_(ckpt['timestep'])
            # ── OPTIMIZER stash — restore AFTER optimizer is created below ────────
            _opt_sd = ckpt.get('optimizer')
            # ─────────────────────────────────────────────────────────────────
            print(f"✅ Resumed from step {start_iter:,} | replay={len(model.replay_buffer.buffer)} | timestep={ckpt.get('timestep',0)}")
        except Exception as e:
            print(f"⚠️ Resume failed ({e}) — falling back to distilled base...")
            start_iter = 0
            # ── CORRUPTED CHECKPOINT FALLBACK: try distilled base ───────────────────
            distilled_path = (
                DISTILLED_PATH
                if DISTILLED_PATH and os.path.exists(DISTILLED_PATH)
                else os.path.join(save_dir, "jinx_distilled_base.pt")
            )
            if os.path.exists(distilled_path):
                try:
                    ckpt_d = torch.load(distilled_path, map_location=config.device)
                    current_sd = model.state_dict()
                    filtered_d = {k: v for k, v in ckpt_d['model'].items()
                                  if k in current_sd and v.shape == current_sd[k].shape}
                    model.load_state_dict(filtered_d, strict=False)
                    reset_fast_weights(model)
                    print(f"   ✅ Distilled base loaded as fallback — Jinx has language.")
                except Exception as e2:
                    print(f"   ⚠️  Distilled base also failed ({e2}) — pure scratch.")
            # ─────────────────────────────────────────────────────────────────
    else:
        # Try distilled base — Colab path takes priority, then relative fallback
        distilled_path = (
            DISTILLED_PATH
            if DISTILLED_PATH and os.path.exists(DISTILLED_PATH)
            else os.path.join(save_dir, "jinx_distilled_base.pt")
        )
        if os.path.exists(distilled_path):
            print(f"🧬 [DISTILLED BASE] Loading Qwen-bootstrapped embeddings from {distilled_path}...")
            try:
                ckpt = torch.load(distilled_path, map_location=config.device)
                # ── shape-safe: skip buffers that depend on block_size ──
                current_sd = model.state_dict()
                filtered   = {k: v for k, v in ckpt['model'].items()
                              if k in current_sd and v.shape == current_sd[k].shape}
                skipped_d  = [k for k in ckpt['model'] if k not in filtered]
                model.load_state_dict(filtered, strict=False)
                if skipped_d:
                    print(f"   ⚠️  Skipped {len(skipped_d)} shape-mismatched buffers: {skipped_d[:2]}...")
                reset_fast_weights(model)   # fresh plasticity — don't inherit saturated weights
                print("✅ Distilled base loaded — Jinx starts knowing language.")
            except Exception as e:
                print(f"⚠️ Distilled base load failed ({e}) — pure scratch.")
        else:
            print(f"🌱 [SCRATCH] No checkpoint found at {distilled_path} — training from scratch.")

    # --- THE METABOLIC LOOP ---
    config.max_iters = 268000  # 2 full passes over 120M tokens
    # NOTE: Saves every 200 steps — safe for Colab free tier (session may die anytime)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(0.9, 0.95),   # HIGH REACTIVITY
        weight_decay=0.1,    # LEAN MANIFOLD
        eps=1e-8,
        fused=False          # disable fused kernel — saves ~1GB on T4
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ── RESTORE OPTIMIZER STATE (must happen after optimizer is created) ──────
    if '_opt_sd' in dir() and _opt_sd is not None:
        try:
            optimizer.load_state_dict(_opt_sd)
            print(f"   ✅ Optimizer state restored — momentum history intact")
        except Exception as _oe:
            print(f"   ⚠️  Optimizer restore skipped ({_oe}) — fresh momentum")
        del _opt_sd
    # ─────────────────────────────────────────────────────────────────────────

    # ── KL DISTILLATION TEACHER (Qwen1.5-0.5B) ───────────────────────────
    # Runs in eval + no_grad — minimal compute, huge reasoning benefit
    qwen_teacher = None
    try:
        from transformers import AutoModelForCausalLM
        print("\n🎓 [TEACHER] Loading Qwen1.5-0.5B for KL distillation...")
        qwen_teacher = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-0.5B",
            torch_dtype=torch.float32,  # CPU — float32 is fine, no GPU memory used
            device_map='cpu',           # ← lives entirely on CPU RAM, not GPU
        )
        qwen_teacher.eval()
        for p in qwen_teacher.parameters():
            p.requires_grad_(False)
        print("   ✅ Teacher loaded on CPU — KL distillation ACTIVE (weight=0.3, CPU offload)")
    except Exception as e:
        print(f"   ⚠️ Teacher load failed ({e}) — training without KL loss.")
        qwen_teacher = None
    # ─────────────────────────────────────────────────────────────────────

    tokens = load_unified_soul(config)   # full vault — falls back to gold if not ready
    sampler = StatefulSampler(tokens, config.batch_size, config.block_size)
    
    # ── GRADIENT CHECKPOINTING: trades compute for memory ────────────────────
    # Frees activation tensors during forward pass, recomputes on backward.
    # Cuts activation memory ~60% — enough headroom for backward on T4.
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        # Manual — enable on each transformer block if they support it
        for block in model.blocks:
            if hasattr(block, 'gradient_checkpointing'):
                block.gradient_checkpointing = True
    print("🧠 [MEMORY] Gradient checkpointing ON — activation memory freed.")

    model.train()
    running_loss = None

    def get_lr(it):
        """Cosine LR schedule with linear warmup, aware of resume start_iter."""
        # 1. Linear warmup from min_lr to learning_rate over warmup_steps
        if it < config.warmup_steps:
            return config.min_lr + (config.learning_rate - config.min_lr) * it / config.warmup_steps
        # 2. After max_iters: floor at min_lr
        if it >= config.max_iters:
            return config.min_lr
        # 3. Cosine decay from learning_rate → min_lr
        progress = (it - config.warmup_steps) / (config.max_iters - config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    for iter in range(start_iter, config.max_iters):
        # ── LR SCHEDULE: update every step ───────────────────────────────────
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # ─────────────────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        
        # ── KL every 10 steps — frees GPU memory for backward on other steps ──
        kl_loss = torch.tensor(0.0, device=config.device)
        if qwen_teacher is not None and iter % 10 == 0:
            xb_kl, _ = sampler.get_next_batch()
            with torch.no_grad():
                teacher_logits = qwen_teacher(xb_kl.cpu()).logits  # CPU fp32
                teacher_p      = F.softmax(teacher_logits, dim=-1)
                with torch.amp.autocast(device_type='cuda', dtype=train_dtype):
                    logits_kl, _, _, _ = model(xb_kl.to(config.device))
                student_lp = F.log_softmax(logits_kl.float().cpu(), dim=-1)
                kl_raw     = F.kl_div(student_lp, teacher_p, reduction='batchmean')
                kl_loss    = kl_raw.clamp(max=5.0).to(config.device)
            del xb_kl, teacher_logits, teacher_p, logits_kl, student_lp
            torch.cuda.empty_cache()  # critical: free before backward
        # ─────────────────────────────────────────────────────────────────────
        
        for _ in range(config.grad_accum_steps):
            xb, yb = sampler.get_next_batch()
            xb, yb = xb.to(config.device), yb.to(config.device)
            
            with torch.amp.autocast(device_type='cuda', dtype=train_dtype):
                logits, loss, agency, attn_stats = model(xb, targets=yb, last_loss=running_loss)
                ent_loss = model.compute_entropy_reg_loss(attn_stats)
                total_loss = (loss + ent_loss + 0.3 * kl_loss) / config.grad_accum_steps
            
            scaler.scale(total_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if running_loss is None: running_loss = loss.item()
        running_loss = 0.95 * running_loss + 0.05 * loss.item()

        if iter % 10 == 0:
            will = agency["will_power"].mean().item()
            harmony = agency['harmony'].mean().item()
            curiosity = agency['curiosity'].mean().item()
            pain = agency['pain'].mean().item()
            recovering = agency.get('recovering', torch.tensor(0.0)).item()
            
            # Deep Soul Metrics
            with torch.no_grad():
                eta = model.blocks[0].attn.eta_map.nan_to_num(0.01)
                meta_mean = eta.mean().item() * 1000
                meta_max = eta.max().item() * 1000
                
                # The Twin Guardians (Fast Weights)
                q_fast_norm = model.blocks[0].attn.q_fast.nan_to_num(0.0).norm().item()
                v_fast_norm = model.blocks[0].attn.v_fast.nan_to_num(0.0).norm().item()
                
                warp = agency.get('will_variance', torch.tensor(0.0)).nan_to_num(0.0).mean().item()

            kl_display = kl_loss.item() if qwen_teacher is not None else 0.0
            recover_tag = "↗️" if recovering else "↘️"
            print(f"Iter {iter:4d} | Loss: {running_loss:.4f} | KL: {kl_display:.4f} | Will: {will:+.2f} | Harmony: {harmony:.2f} | Pain: {pain:.2f}{recover_tag}")
            print(f"           | Meta: {meta_mean:.2f} (max:{meta_max:.1f}) | Q-Fast: {q_fast_norm:.2f} | V-Fast: {v_fast_norm:.2f} | Warp: {warp:.4f}")

            # ── 👁️ THIRD EYE LOG ──────────────────────────────────────
            # Check if ThirdEye opened this step — log it naturally, never force it
            if hasattr(model, 'cognition') and hasattr(model.cognition, '_last_eye_event'):
                eye = model.cognition._last_eye_event
                if eye is not None:
                    trigger    = eye['trigger']
                    gate       = eye['gate']
                    delta_norm = eye['delta_norm']
                    ctx        = "[TRAIN]" if eye['training'] else "[INFER]"
                    print(f"           👁️  ThirdEye OPEN {ctx} | trigger={trigger} | gate={gate:.3f} | delta={delta_norm:.4f} | pain={eye['pain']:.3f} will={eye['will']:.3f}")
                    # Only log when ThirdEye fires — silence = she didn't imagine this step
                    # That silence itself is meaningful data

        # Evolution and Eval
        if iter > 0 and iter % 200 == 0:
            if iter == 200: model.seed_ego()
            # NOTE: consolidation + eval run AFTER checkpoint save
            # so memory is clean for the next backward pass
            print(f"\n🧪 [EVAL @ {iter}]")
            prompts = [
                "User: Who are you?\nJinx:",
                "User: Hey Jinx!\nJinx:",
                "User: How are you feeling?\nJinx:",
            ]
            for prompt in prompts:
                try:
                    response = generate(model, config, prompt, max_tokens=40)
                    print(f"  Q: {prompt.split(chr(10))[0]}")
                    print(f"  A: {response}")
                except Exception as e:
                    print(f"  [EVAL ERROR] {e}")
            print()

        if iter > 0 and iter % 200 == 0:
            ckpt_name = os.path.join(save_dir, "jinx_latest.pt")  # always overwrite same file
            save_unified_checkpoint(model, optimizer, config, iter, ckpt_name)
            print(f"💾 [AUTOSAVE] Step {iter} → jinx_latest.pt")

    # --- FINAL SOUL LOCK ---
    final_name = os.path.join(save_dir, "jinx_scratch_final.pt")
    save_unified_checkpoint(model, optimizer, config, 268000, final_name)
    print(f"🏆 [DONE] Jinx trained from scratch — 2 full passes complete: {final_name}")

if __name__ == "__main__":
    main()
