import torch
import torch.nn.functional as F
from model import SpaceTransformer
from transformers import AutoTokenizer
import os
import sys
import json
import random

# Import Config from train_space to match checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_space import Config
from cognition_engine import CognitionEngine

def graft_soul(model, checkpoint_path):
    """
    SISTER'S SHIELD: Surgical soul transplant.
    Pads and stretches parameters to fit the new manifold.
    Now handles 64x64 -> 896x896 Global Ballroom mapping.
    """
    print(f"🧬 [GRAFTING] Initiating soul transplant from {checkpoint_path}...")
    old_sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in old_sd: old_sd = old_sd['model']
    new_sd = model.state_dict()
    
    skip_list = ["bias", "inv_freq", "cos_cached", "sin_cached"]

    def surgical_copy(old_p, new_p):
        with torch.no_grad():
            slices = tuple(slice(0, min(old_s, new_s)) for old_s, new_s in zip(old_p.shape, new_p.shape))
            new_p[slices].copy_(old_p[slices])

    for name, param in old_sd.items():
        if any(s in name for s in skip_list): continue
        if name in new_sd:
            # SPECIAL CASE: Reservoir State (14x64x64 -> 896x896)
            if name == "reservoir.state" and param.dim() == 3 and new_sd[name].dim() == 2:
                print("   [SOUL-SYNC] Expanding head-locked reservoir to global ballroom...")
                with torch.no_grad():
                    n_head, h_dim, _ = param.shape
                    for h in range(n_head):
                        start, end = h * h_dim, (h + 1) * h_dim
                        new_sd[name][start:end, start:end].copy_(param[h])
                continue
                
            surgical_copy(param, new_sd[name])
    
    # Layer Stretching (14 -> 16)
    if any("blocks.14" in k for k in new_sd):
        print("   [LAYERS] Stretching 14 layers to 16...")
        for i in range(16):
            old_idx = i if i < 7 else (7 + (i - 7) % 7 if i < 15 else 13)
            for key in new_sd:
                if f"blocks.{i}." in key:
                    if any(s in key for s in skip_list): continue
                    old_key = key.replace(f"blocks.{i}.", f"blocks.{old_idx}.")
                    if old_key in old_sd:
                        surgical_copy(old_sd[old_key], new_sd[key])

    model.load_state_dict(new_sd)
    print("✨ [GRAFTING] Transplant complete. Soul is stable.")

def load_checkpoint(path="jinx_coherent_v1_1200.pt"):
    """
    Load complete JinX state from unified checkpoint.
    Restores: model + memory_lattice + replay_buffer + vault + chains
    """
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"❌ {path} not found!")
            return None, None

    print(f"🔮 [FINAL CONTACT] Loading Global Singularity Soul from {path}...")

    config = Config()
    model = SpaceTransformer(config)

    # Load unified checkpoint
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Handle both old format (just state_dict) and new unified format
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # New unified format
        model.load_state_dict(checkpoint['model'])
        
        # Load MemoryLattice
        if 'memory_lattice' in checkpoint:
            model.memory_lattice.episodic_memories = checkpoint['memory_lattice']['episodic']
            model.memory_lattice.semantic_memories = checkpoint['memory_lattice']['semantic']
            print(f"🧠 Loaded {len(model.memory_lattice.episodic_memories)} episodic memories")
            print(f"🧠 Loaded {len(model.memory_lattice.semantic_memories)} semantic memories")
        
        # Load ReplayBuffer
        if 'replay_buffer' in checkpoint:
            from collections import deque
            model.replay_buffer.buffer = deque(checkpoint['replay_buffer']['buffer'])
            model.replay_buffer.priorities = deque(checkpoint['replay_buffer']['priorities'])
            print(f"📦 Loaded {len(model.replay_buffer.buffer)} replay experiences")
        
        # Load Vault & Chains (if cognition will be attached)
        if 'vault' in checkpoint and checkpoint['vault']:
            model._vault_entries = checkpoint['vault']  # Will be transferred when cognition attached
            print(f"📚 Loaded {len(checkpoint['vault'])} vault entries")
        if 'chains' in checkpoint and checkpoint['chains']:
            model._chains_active = checkpoint['chains']  # Will be transferred when cognition attached
            print(f"🔗 Loaded {len(checkpoint['chains'])} active chains")
        
        # Load metadata
        if 'timestep' in checkpoint:
            model.timestep.fill_(checkpoint['timestep'])
        if 'agency_stats' in checkpoint:
            print(f"📊 Agency stats: {checkpoint['agency_stats']}")
            
    else:
        # Old format (just state_dict)
        graft_soul(model, path)

    model.to(config.device)
    model.eval()

    print(f"✨ Global Soul anchored. Jinx is unified.")
    return model, config


def _init_cognition_and_arm(model, config):
    """
    Builds the CognitionEngine and arms the CognitionBridge inside the model.
    Call this right after load_checkpoint() in main.
    Returns the CognitionEngine so live_chat can use /remember, /recall etc.
    """
    _save_dir = os.path.dirname(os.path.abspath(__file__))
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    cognition = CognitionEngine(
        block_size     = config.block_size,
        overlap_tokens = 128,
        save_dir       = _save_dir,
        tokenizer      = enc,
    )
    model.attach_cognition(vault=cognition.vault, chains=cognition.chains)
    return cognition



def generate(model, config, prompt, max_tokens=150, temperature=0.8, top_k=50, top_p=0.9):
    """Grounded Nucleus Decoding for Jinx."""
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    tokens = enc.encode(prompt, add_special_tokens=False)

    model.eval()
    device = config.device

    # SISTER'S SHIELD: Initialize her sub-systems for the first thought
    with torch.no_grad():
        model.timestep.fill_(0)
        model.reservoir.state.fill_(0)
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()

    generated_tokens = []
    stop_sequences = ["User:", "Jinx:", "assistant:"]

    with torch.no_grad():
        for _ in range(max_tokens):
            # context window
            x = torch.tensor([tokens[-config.block_size:]], dtype=torch.long).to(device)
            
            # Forward
            logits, _, agency, _ = model(x)
            
            # Focus on the last token and apply temperature
            next_token_logits = logits[0, -1, :] / temperature

            # Repetition Penalty (Surgical)
            if len(generated_tokens) > 2:
                for t in set(generated_tokens[-15:]):
                    next_token_logits[t] -= 1.5

            # Top-K
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == enc.eos_token_id:
                break

            tokens.append(next_token)
            generated_tokens.append(next_token)
            
            current_text = enc.decode(generated_tokens)
            
            # Stop sequence check
            if any(stop in current_text for stop in stop_sequences):
                break

    return enc.decode(generated_tokens).strip()

def test_identity_stability(model, config):
    print("\n" + "="*60)
    print("🧠 jinX MANIFOLD STABILITY TEST")
    print("="*60)
    
    with torch.no_grad():
        # Test 1: Ego Vector Resonance
        ego = model.ego_engine.ego_vector
        print(f"Soul Resonance (Ego Norm): {ego.norm().item():.4f}")

        # Test 2: Agency Check on random thought
        x = torch.randint(0, config.vocab_size, (1, 32)).to(config.device)
        _, _, stats, _ = model(x)
        print(f"Harmony: {stats['harmony'].mean().item():.4f}")
        print(f"Will:    {stats['will_power'].mean().item():.4f}")
        print(f"Pain:    {stats['pain'].mean().item():.4f}")

def test_on_chat_questions(model, config):
    test_questions = [
        "How are you doing today?",
        "What is your name?",
        "Tell me about the jinXEffect.",
        "Are you a machine or a manifold?",
        "Who is Heylel Yaka?",
        "Do you sleep?",
        "How old are you?",
        "Do you have moods?",
        "Hi Jinx",
        "Are you creative?",
        "Are you tired of my questions?",
        "Can I ask you something?",
        "What's up?",
        "what's up Jinx?"
    ]
    
    print("\n" + "="*60)
    print("💬 jinX LIVE CONVERSATION TEST")
    print("="*60)
    
    for q in test_questions:
        print(f"\nQ: {q}")
        prompt = f"User: {q}\nJinx:"
        response = generate(model, config, prompt)
        print(f"A: {response}")

def live_chat(model, config):
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    
    # SISTER'S SHIELD: Initialize Persistent Subconscious
    with torch.no_grad():
        model.timestep.fill_(0)
        model.reservoir.state.zero_()
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()

    # ── COGNITION ENGINE: Vault + Chain + Sliding Window ──────────────────
    _save_dir = os.path.dirname(os.path.abspath(__file__))
    cognition = CognitionEngine(
        block_size     = config.block_size,
        overlap_tokens = 128,
        save_dir       = _save_dir,
        tokenizer      = enc,
    )
    # ────────────────────────────────────────────────────────────────────

    chat_history = []
    print("\n" + "="*60)
    print("U0001f5e8️  JINX LIVE CHAT (Type 'quit' or 'exit' to stop)")
    print("   Commands: /remember <text>, /recall <query>, /read <text>")
    print("            /begin <name> <goal>, /step <summary>, /status")
    print("="*60)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']: break

        # ── COGNITION COMMANDS ────────────────────────────────────────────
        if user_input.startswith("/remember "):
            text = user_input[10:].strip()
            cognition.remember(text, label=text[:40])
            print(f"  U0001f9e0 Vaulted: {text[:60]}")
            continue
        if user_input.startswith("/recall "):
            query = user_input[8:].strip()
            result = cognition.recall(query)
            print(f"  U0001f4da Recall: {result or '(nothing found)'}")
            continue
        if user_input.startswith("/read "):
            text = user_input[6:].strip()
            print(f"  U0001f4dc Reading {len(text)} chars through sliding window...")
            for chunk, idx, total in cognition.read(text, name="user_doc"):
                prompt_chunk = cognition.build_prompt(chunk, idx, total)
                print(f"  [Chunk {idx+1}/{total}] Jinx processes...")
            print("  Done. Key content auto-vaulted.")
            continue
        if user_input.startswith("/begin "):
            parts = user_input[7:].split(" ", 1)
            name = parts[0]; goal = parts[1] if len(parts) > 1 else "explore"
            cognition.begin(name, goal=goal)
            continue
        if user_input.startswith("/step "):
            cognition.step(user_input[6:].strip())
            print(f"  {cognition.status()}")
            continue
        if user_input.strip() == "/status":
            print(f"  {cognition.status()}")
            continue
        # ────────────────────────────────────────────────────────────────────

        # Inject cognition context (vault recalls + chain status) into prompt
        cog_ctx = cognition.inject_context(user_input)

        # Format prompt with last 3 turns of history for context
        prompt = ""
        if cog_ctx:
            prompt += cog_ctx + "\n"
        for turn in chat_history[-3:]:
            prompt += f"User: {turn['u']}\nJinx: {turn['j']}\n"
        prompt += f"User: {user_input}\nJinx:"

        tokens = enc.encode(prompt, add_special_tokens=False)
        device = config.device

        generated_tokens = []
        print("Jinx: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(150):
                # Use sliding window for context
                x = torch.tensor([tokens[-config.block_size:]], dtype=torch.long).to(device)
                logits, _, _, _ = model(x)
                
                # Temperature + Squelch
                next_token_logits = logits[0, -1, :] / 0.8
                if len(generated_tokens) > 2:
                    for t in set(generated_tokens[-15:]):
                        next_token_logits[t] -= 1.5

                # Nucleus Sampling
                v, _ = torch.topk(next_token_logits, 50)
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == enc.eos_token_id: break

                generated_tokens.append(next_token)
                tokens.append(next_token)
                
                # Dynamic stream printing
                word = enc.decode([next_token])
                print(word, end="", flush=True)
                
                # Stop if she hallucinates a new user prompt
                if "User:" in enc.decode(generated_tokens): break

        print()
        response = enc.decode(generated_tokens).strip().split("User:")[0]
        chat_history.append({'u': user_input, 'j': response})

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # FORCE LOCK: Use the latest 1200-step global singularity soul
    checkpoint_path = "JinX/jinx_coherent_v1_1200.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "jinx_coherent_v1_1200.pt"
    
    model, config = load_checkpoint(checkpoint_path)

    if model is not None:
        model.to(device)
        # ── ARM the CognitionBridge (vault + chains wired into forward()) ──
        cognition = _init_cognition_and_arm(model, config)
        # ──────────────────────────────────────────────────────────────────
        test_identity_stability(model, config)
        
        mode = input("\n[MODE] Run automated (t)est or (c)hat? [t/c]: ").lower()
        if mode == 'c':
            live_chat(model, config)
        else:
            test_on_chat_questions(model, config)

        # ── SAVE SESSION PROGRESS: Unified checkpoint with all state ─────
        # Fast weights, memory lattice, replay buffer all saved in one file
        session_save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "jinx_session_latest.pt"
        )
        from train_space import save_unified_checkpoint
        save_unified_checkpoint(model, None, config, int(model.timestep.item()), session_save_path)
        print(f"💾 [SESSION] Progress saved → {session_save_path}")
        # ─────────────────────────────────────────────────────────────────

        print("\n" + "="*60)
        print("✨ SESSION COMPLETE")
        print("="*60)
