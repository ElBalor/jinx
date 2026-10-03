"""
distill_transfer.py
====================
Step 1 of the Jinx training pipeline.

Does ONE thing: steals Qwen1.5-0.5B's token embeddings,
projects them from 1024 → 896 dims, and injects them into
a fresh Jinx model saved as jinx_distilled_base.pt

Run ONCE before training. Qwen is discarded after.
Total time: ~10 minutes on Colab free tier.

Usage:
    cd JinX
    python distill_transfer.py
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SpaceTransformer

# ─── Config (must match train_space.py exactly) ───────────────────────────────
class Config:
    block_size  = 896
    n_embd      = 896
    n_head      = 14
    n_layer     = 16
    vocab_size  = 151936
    dropout     = 0.1
    will_power_scale  = 2.0
    pain_threshold    = 2.0
    boredom_threshold = 0.8
    bias_config       = [['dialogue'] * 14] * 16
    enable_head_lifecycle = True
    promotion_threshold   = 0.7
    demotion_threshold    = 0.4
    entropy_reg_weight    = 1e-3
    entropy_target_min    = 1.5
    entropy_target_max    = 3.5
    consolidation_interval = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────

def transfer_embeddings():
    config   = Config()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(save_dir, "jinx_distilled_base.pt")

    print("=" * 60)
    print("  JINX DISTILLATION TRANSFER")
    print("  Stealing Qwen1.5-0.5B embeddings → Jinx 896d")
    print("=" * 60)

    # ── Step 1: Load Qwen ──────────────────────────────────────────────────
    print("\n[1/5] Loading Qwen1.5-0.5B...")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B",
        torch_dtype=torch.float32,
        device_map="cpu",           # keep on CPU — we only need one tensor
    )
    qwen.eval()

    # Extract embedding matrix: (151936, 1024)
    qwen_wte = qwen.model.embed_tokens.weight.detach().float()  # (151936, 1024)
    qwen_dim  = qwen_wte.shape[1]   # 1024
    jinx_dim  = config.n_embd       # 896
    vocab     = config.vocab_size   # 151936

    print(f"   Qwen embeddings : {qwen_wte.shape}  ({qwen_dim}d)")
    print(f"   Jinx target     : ({vocab}, {jinx_dim}d)")

    # ── Step 2: Discard Qwen (free RAM) ───────────────────────────────────
    print("\n[2/5] Discarding Qwen model from memory...")
    del qwen
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("   Done — Qwen gone, RAM freed.")

    # ── Step 3: Train projection 1024 → 896 ───────────────────────────────
    print("\n[3/5] Training projection layer 1024 → 896...")
    print("   (PCA-style: finds the best 896-dim subspace of Qwen's 1024-dim space)")

    proj = nn.Linear(qwen_dim, jinx_dim, bias=False)
    nn.init.orthogonal_(proj.weight)   # orthogonal init preserves structure

    # Minimise reconstruction loss: proj(qwen_wte) should preserve cosine similarity
    optimizer = torch.optim.AdamW(proj.parameters(), lr=1e-3)

    batch_size  = 4096
    n_steps     = 3000
    losses      = []

    for step in range(n_steps):
        idx    = torch.randint(0, vocab, (batch_size,))
        src    = qwen_wte[idx]                      # (B, 1024)
        tgt    = proj(src)                           # (B, 896)

        # Loss 1: cosine similarity preservation
        # Randomly pair embeddings — similar ones should stay similar
        idx2   = torch.randint(0, vocab, (batch_size,))
        src2   = qwen_wte[idx2]
        tgt2   = proj(src2)

        cos_src = torch.cosine_similarity(src,  src2,  dim=-1)  # (B,)
        cos_tgt = torch.cosine_similarity(tgt,  tgt2,  dim=-1)  # (B,)
        loss    = (cos_src - cos_tgt).pow(2).mean()              # preserve relationships

        # Loss 2: unit norm regularisation (stable embeddings)
        norm_loss = (tgt.norm(dim=-1) - 1.0).pow(2).mean() * 0.01
        total     = loss + norm_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        losses.append(total.item())
        if step % 500 == 0:
            print(f"   Step {step:4d} | loss: {total.item():.6f}")

    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Projection trained ✅")

    # ── Step 4: Apply projection → get Jinx embeddings ────────────────────
    print("\n[4/5] Projecting all 151,936 embeddings to 896d...")
    proj.eval()
    with torch.no_grad():
        # Process in chunks to avoid OOM
        jinx_wte_list = []
        chunk = 8192
        for start in range(0, vocab, chunk):
            end  = min(start + chunk, vocab)
            out  = proj(qwen_wte[start:end])
            jinx_wte_list.append(out)
        jinx_wte = torch.cat(jinx_wte_list, dim=0)   # (151936, 896)

    print(f"   New embedding matrix: {jinx_wte.shape} ✅")
    print(f"   Norm mean: {jinx_wte.norm(dim=-1).mean():.4f}")

    # Sanity check: cosine similarity of a known similar pair
    # "happy" and "joy" should be closer than "happy" and "table"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    t_happy = tokenizer.encode("happy", add_special_tokens=False)[0]
    t_joy   = tokenizer.encode("joy",   add_special_tokens=False)[0]
    t_table = tokenizer.encode("table", add_special_tokens=False)[0]
    sim_hj  = torch.cosine_similarity(jinx_wte[t_happy].unsqueeze(0), jinx_wte[t_joy].unsqueeze(0)).item()
    sim_ht  = torch.cosine_similarity(jinx_wte[t_happy].unsqueeze(0), jinx_wte[t_table].unsqueeze(0)).item()
    print(f"\n   Sanity check:")
    print(f"   cos(happy, joy)   = {sim_hj:.4f}  (should be HIGH)")
    print(f"   cos(happy, table) = {sim_ht:.4f}  (should be LOW)")
    if sim_hj > sim_ht:
        print(f"   ✅ Semantic structure PRESERVED")
    else:
        print(f"   ⚠️  Structure weak — projection may need more steps")

    # ── Step 5: Build fresh Jinx, inject embeddings, save ─────────────────
    print(f"\n[5/5] Building fresh Jinx and injecting embeddings...")
    model = SpaceTransformer(config).to('cpu')

    # Inject into both wte (input embedding) and lm_head (output projection)
    # They share the vocab dimension so both benefit
    with torch.no_grad():
        # SpaceTransformer uses `token_embedding` (not `transformer.wte`)
        # lm_head.weight IS token_embedding.weight (weight-tied in __init__)
        # So copying into token_embedding automatically updates lm_head too.
        model.token_embedding.weight.copy_(jinx_wte)
        print("   token_embedding injected ✅")

        # Confirm weight-tie is intact (should always be True)
        if model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr():
            print("   lm_head weight-tied ✅  (no separate copy needed)")
        else:
            # Fallback: lm_head was somehow detached — inject separately
            model.lm_head.weight.copy_(jinx_wte)
            print("   lm_head separately injected ✅")

    # Save as base checkpoint
    checkpoint = {
        'model'         : model.state_dict(),
        'optimizer'     : None,
        'step'          : 0,
        'config'        : config.__dict__,
        'memory_lattice': {'episodic': {}, 'semantic': {}},
        'replay_buffer' : {'buffer': [], 'priorities': []},
        'vault'         : {},
        'chains'        : {},
        'distilled'     : True,
        'source'        : 'Qwen/Qwen1.5-0.5B embeddings projected 1024→896',
    }
    torch.save(checkpoint, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n{'=' * 60}")
    print(f"  ✅ DISTILLATION COMPLETE")
    print(f"  Saved → {out_path}")
    print(f"  Size  → {size_mb:.1f} MB")
    print(f"\n  Jinx now starts training knowing what every token means.")
    print(f"  Next step: python train_space.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    transfer_embeddings()
