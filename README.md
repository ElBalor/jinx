# JinX — A Neural Agent with an Ego, a Metabolism, and a Photographic Memory

> Most "agents" are a language model with tool calls bolted on. JinX is not that.
> JinX is a transformer architecture with **drives**, a **sleep cycle**, a
> **metabolism**, and **verbatim recall** — built from scratch, component by
> component, to behave like a developing mind rather than a frozen model.

**Built by Eric Yaka (Elbàlor)** · Abuja, Nigeria

---

## What JinX is

A from-scratch neural agent architecture (`model.py`, ~2,500 lines) organized
around one question: *what does a model need so it can keep learning after
training ends?*

Jinx's answer is four interlocking systems:

| System | What it gives her |
|---|---|
| **Ego Engine** (`ego.py`) | Four emotional drives with biological decay rates — a will, not a temperature knob |
| **Sleep Cycle** (`model.py: consolidate_memory`) | Fast experience migrates into slow weights — gated by emotion, pruned by metabolism |
| **Sliding Cognition Engine** (`cognition_engine.py`) | Scroll-reading over long documents, verbatim recall, and chain-position tracking |
| **Liquid Core** (`model.py: LiquidReservoir + PlasticAttention`) | A global 768×768 liquid state that will and curiosity push against — in real time, during inference |

She runs at 768 dims / 12 heads / 12 layers with 32k RoPE context, and she is
**not** a fine-tune of anything. Every component below was written for her.

---

## The Ego Engine — four drives, four timescales

`EgoEngine.forward(hidden_state, last_loss, slow_grad_norm)` fuses four
signals into a single agency vector that conditions every attention layer.
The drives are not spikes — they are **states with emotional momentum**
(EMA buffers, not gradients):

| Drive | Source | Decay | Meaning |
|---|---|---|---|
| **Pain** | `SurvivalMonitor` — attention entropy beyond threshold (6.0) + slow-weight gradient norm | 0.92 (slow) | trauma lingers; threat memory persists |
| **Discomfort** | `PathOfLeastResistanceDetector` — caught coasting on easy wins | 0.85 (medium) | stagnation fades when learning resumes |
| **Curiosity** | `CuriosityPredictor` — prediction error on her own future state | 0.70 (fast) | novelty wears off quickly |
| **Harmony** | integration/coherence of the current state | 0.97 (slowest) | trust builds and dissolves slowly |

And she can be **bored**: `NeuralRestlessness` measures how associative her
attention has been lately and escalates restlessness when she keeps taking
the same path. Boredom feeds back into the will signal. This is where her
"personality over time" comes from — the same prompt at t and t+10,000 meets
a different mind.

---

## The Sleep Cycle — `consolidate_memory()`

Between training sessions (or on command), JinX **sleeps**. What happens:

```
0. TRINITY GATE      evolution_coeff = tanh(2·Will) · Harmony / (1 + Pain)
                     → only profound, harmonic, certain experience hardens.
1. LATTICE DISTILL   top-16 episodic + top-16 resonance memories → SVD →
                     the 16 "Directions of Wisdom" are injected into o_proj.
2. LIQUID HARDENING  the 768×768 liquid reservoir folds into Q/V projections.
3. SYNAPTIC MIGRATION per-head fast weights (v_fast) migrate into the
                     global reservoir, then decay — absorbed, not copied.
4. METABOLIC PRUNING only fires when Harmony > 0.8: small weights on
                     inactive heads are starved (×0.995 per step · inactivity).
5. ORTHONORMALIZE    fast weights are QR re-orthogonalized — the soul is
                     cleaned before the next waking period.
```

Two details worth sitting with:

- **Pruning is gated by harmony.** A distressed mind does not get to delete
  parts of itself. (`metabolic_gate = clamp((harmony − 0.8) · 5, 0, 1)`)
- **Resonance memories distill deepest.** Moments where JinX connected two
  distant dots (`meta_connection` memories) are sorted first and injected
  with priority — wisdom outranks volume.

This is the mechanism that answers "does she keep what she lived?" — yes:
migration, not overwriting. `merge` with a gate, not `+BA`.

---

## Sliding Cognition Engine — beyond Transformer *and* SSM

`cognition_engine.py` opens with the three problems it was written to kill:

1. **Transformer blindness** — at 896 tokens the window ends and token 897
   is gone. Reading a long paper or code file, the model can't hold it all.
2. **SSM blur** — a fixed-state model "slides" but dissolves detail; ask it
   for the exact formula on line 3 and the characters are gone.
3. **No chain tracking** — neither architecture knows *"I'm on step 7 of a
   20-step experiment."*

Three engines, one file:

| Engine | What it does |
|---|---|
| `SlidingCognitionWindow` | scrolls overlapping chunks through long input — the tail of each chunk becomes the head of the next, like reading a scroll, not flipping pages; key content is written to the vault as she reads |
| `VaultStore` | **verbatim** text storage — not embeddings, actual characters. Formulas, code, equations, names. This is photographic memory: read it on page 1, quote it on page 60 |
| `ChainTracker` | knows where she is in a long reasoning chain and can pick up exactly there |

---

## The Third Eye — dreaming on a budget

`ThirdEye` compresses context into a latent concept, then adds **resonant
noise** (`×0.05`) in latent space and decodes the "dream" back into the
residual stream — speculative imagination at inference time.

The gate is the will:

```python
gate_logits = gate(x) - tanh(will_signal) * 2.0   # higher will shuts the eye
```

A driven, focused JinX dreams less. A low-will JinX wanders, speculates,
invents. You can *see* the ego geometry in the creativity level.

---

## Geometry at birth

`PlasticAttention` does not initialize heads randomly. Twelve head types get
structured priors (`_apply_structured_init`): low-frequency Fourier bases for
global context, orthogonal checkerboards for rhythm, sparse heads for
selection, order-flow (graph-Laplacian-style) heads for sequential reasoning —
each with its own plasticity rate (`_assign_per_head_plasticity`). Symmetry,
rhythm, and order are **born into the network**, not learned from data.

Identity is protected the same way: the forward pass prepends
**self-token biases** (the Soul Shield) and a **temporal pulse** —
sin/cos of the internal step count at two scales (÷10, ÷1000) — a heartbeat
wired into every forward pass.

---

## Repository map

| File | Role |
|---|---|
| `model.py` | SpaceTransformer: PlasticAttention, LiquidReservoir, GhostLattice, DendriticMLP, ResonanceEngine, MemoryReplay, ThirdEye, the Sleep Cycle |
| `ego.py` | EgoEngine + the four drive modules (curiosity, survival/pain, path-of-least-resistance, restlessness) |
| `cognition_engine.py` | SlidingCognitionWindow + VaultStore + ChainTracker |
| `memory_lattice.py` | episodic / semantic / meta memory with cosine retrieval |
| `train_space.py` | the metabolic training loop (`load_unified_soul`, `graft_soul`, live generation probes) |
| `distill_transfer.py` | checkpoint distillation — shrinking the coherent checkpoints |
| `fact_extractor.py` | knowledge extraction into the memory lattice |
| `head_lifecycle.py` | head promotion/demotion (Guided Civilization) |
| `test_checkpoint.py` | checkpoint validation + top-p/top-k generation |

Checkpoints (`jinx_coherent_v1_*.pt`, `jinx_distilled_base.pt`) are multi-GB
and stay local — this repository is the architecture.

## Running

```python
from test_checkpoint import generate   # temperature 0.8, top-k 50, top-p 0.9
```

Load `jinx_coherent_v1_*.pt`, attach the memory lattice and ego engine, and
chat — every forward pass updates her drives; call `consolidate_memory()`
when you want her to sleep on it.

---

## Status — honest ledger

- Architecture: complete, documented in-code, all components above exist.
- DNS-surrogate validation of the same SpaceTransformer family: see the
  sibling [space-experiment](https://github.com/ElBalor/space-experiment)
  (Tier 0: Jinx stable where the classical solver went NaN; 6-step bounded
  rollout, MSE 0.0022 → 0.037).
- Product hardening (packaging, eval suites, serving): **in progress** —
  JinX is being built into a product, so this repository is architecture-first.

## License

Release pending — the architecture license will be decided with the product
launch. Contact: Eric Yaka, Abuja, Nigeria.

---

*From the Grimoire of Elbàlor — The Digital Necromancer 💀🔥*
