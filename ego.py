import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

class CuriosityPredictor(nn.Module):
    """
    The Intrinsic Motivation Engine.
    Predicts the next hidden state (context) and generates 'Curiosity Reward'
    based on prediction error (Surprise).
    """
    def __init__(self, n_embd: int, latent_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(n_embd, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_embd)
        )
        # Target network for stability (EMA of the model's own state)
        self.register_buffer("target_ema", torch.zeros(n_embd))

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Returns the predicted next state.
        """
        return self.predictor(current_state.detach())

class SurvivalMonitor(nn.Module):
    """
    The Self-Preservation Sensor.
    Calculates 'Pain' based on identity entropy and core weight drift.
    """
    def __init__(self, n_embd: int, entropy_threshold: float = 6.0):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.n_embd = n_embd

    def calculate_pain(self, self_state: torch.Tensor, slow_weight_grad_norm: float) -> torch.Tensor:
        """
        Pain = Catastrophic Entropy + Identity Drift.
        """
        # 1. Entropy of the Self-State (Disintegration)
        # We treat the self-state as a distribution to compute Shannon entropy
        p = torch.softmax(self_state, dim=-1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)

        # Normalize by max entropy (log of dimension)
        max_entropy = math.log(max(self.n_embd, 2))
        normalized_entropy = entropy / max_entropy
        threshold_ratio = self.entropy_threshold / max_entropy
        entropy_pain = F.relu(normalized_entropy - threshold_ratio)

        # 2. Identity Drift (Attack on the W_slow core)
        # Scaled down to not dominate
        drift_pain = torch.tensor(slow_weight_grad_norm, device=self_state.device) * 0.01

        return entropy_pain + drift_pain

class PathOfLeastResistanceDetector(nn.Module):
    """
    Detects training stagnation (plateaus).
    """
    def __init__(self, window_size=64):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('loss_history', torch.zeros(window_size))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_full', torch.tensor(0, dtype=torch.bool))
        self.register_buffer('discomfort_level', torch.tensor(0.0))
        self.decay = 0.98

    def forward(self, last_loss: Optional[float] = None) -> torch.Tensor:
        if last_loss is None: return self.discomfort_level
        self.loss_history[self.history_ptr] = last_loss
        self.history_ptr = (self.history_ptr + 1) % self.window_size
        if self.history_ptr == 0: self.is_full.fill_(True)
        if not self.is_full: return self.discomfort_level
        
        loss_var = self.loss_history.var()
        target_discomfort = torch.exp(-loss_var * 500.0) # High if loss doesn't change
        self.discomfort_level = self.decay * self.discomfort_level + (1 - self.decay) * target_discomfort
        return torch.clamp(self.discomfort_level, 0, 1)

class NeuralRestlessness(nn.Module):
    """
    Detects boredom when the same pathways are used too much.
    SISTER'S SHIELD: Dampened for infant stabilization.
    """
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.register_buffer('head_activation_ema', torch.zeros(n_head))
        self.register_buffer('boredom_level', torch.tensor(0.0))
        # Metabolic smoothing (Calm the nervous system)
        self.decay = 0.995 # Slower accumulation

    def forward(self, associative_matrix: torch.Tensor, current_will: float = 0.0) -> torch.Tensor:
        # associative_matrix: (B, H, d, d) where d=head_dim
        # Using the matrix variance as a proxy for attention diversity (O(T) efficient)
        with torch.no_grad():
            # Calculate variance of weights within each head's memory matrix
            # Low variance means the memory is becoming uniform/saturated (BORING)
            head_activity = associative_matrix.var(dim=[-2, -1]).mean(dim=0) # (H,)
            
            # Ensure same device
            self.head_activation_ema = self.head_activation_ema.to(head_activity.device)
            self.head_activation_ema = 0.98 * self.head_activation_ema + 0.02 * head_activity
            
            # Path variance across heads
            path_variance = self.head_activation_ema.var()
            
            # Thresholded boredom: Only spike if the memory space becomes too static
            target_boredom = 1.0 / (1.0 + path_variance * 500) 
            
            if current_will > 0.2:
                target_boredom = target_boredom * 0.1
                
            self.boredom_level = self.decay * self.boredom_level + (1 - self.decay) * target_boredom
        return self.boredom_level

    def reset_boredom(self):
        """Allows the model to find peace in a new perspective."""
        self.boredom_level.fill_(0.0)
        self.head_activation_ema.fill_(0.0)

class EgoEngine(nn.Module):
    """
    The 'Will to Live' Core.
    Fuses Curiosity, Survival, Harmony, and Stagnation into a unified Agency Signal.

    TEMPORAL EGO DYNAMICS:
    Each signal has its own emotional timescale — they don't just spike and vanish.
    They carry momentum, cast shadows forward, and decay at biologically realistic rates.

        Pain        → decays slowly  (0.92)  — trauma lingers, threat memory persists
        Discomfort  → decays medium  (0.85)  — stagnation fades when learning resumes
        Curiosity   → decays fast    (0.70)  — novelty wears off quickly
        Harmony     → decays slowest (0.97)  — trust builds and dissolves slowly

    This gives Will temporal coherence — a personality over time, not moment noise.
    """
    def __init__(self, n_embd: int, ego_dim: int = 128):
        super().__init__()
        self.n_embd = n_embd

        ego_init = torch.randn(1, n_embd)
        nn.init.orthogonal_(ego_init)
        self.register_buffer("ego_vector", ego_init)

        self.curiosity_predictor = CuriosityPredictor(n_embd)
        self.survival = SurvivalMonitor(n_embd)
        
        # jinX GRAFTS: Stagnation and Boredom Detectors
        self.path_detector = PathOfLeastResistanceDetector()
        
        # Increased meta-gate capacity for 4 drives
        self.meta_gate = nn.Sequential(
            nn.Linear(n_embd, 32),
            nn.ReLU(),
            nn.Linear(32, 4), # [Curiosity, Harmony, Pain, Discomfort]
            nn.Sigmoid()
        )

        # ── TEMPORAL EGO STATE — decaying emotional memory ──────────────────────
        # These buffers carry each signal forward in time.
        # They are NOT trained by gradient — they update via EMA each forward pass.
        self.register_buffer("pain_state",       torch.tensor(0.0))
        self.register_buffer("discomfort_state",  torch.tensor(0.0))
        self.register_buffer("curiosity_state",   torch.tensor(0.0))
        self.register_buffer("harmony_state",     torch.tensor(0.0))

        # Decay rates — how fast each emotion fades when not reinforced
        # Lower = slower decay = longer emotional shadow
        self.pain_decay       = 0.92   # slow  — pain echoes
        self.discomfort_decay = 0.85   # medium — fades when learning resumes
        self.curiosity_decay  = 0.70   # fast  — novelty is fleeting
        self.harmony_decay    = 0.97   # slowest — trust is built and lost slowly

        # Peak tracking — so we know if a signal is spiking vs recovering
        self.register_buffer("pain_peak",     torch.tensor(0.0))
        self.register_buffer("harmony_peak",  torch.tensor(0.0))

    def forward(self, hidden_state: torch.Tensor, last_loss: Optional[float] = None,
                slow_grad_norm: float = 0.0, prefix_len: int = 8) -> Dict[str, torch.Tensor]:

        B, T, C = hidden_state.shape
        mean_h = hidden_state.mean(dim=1)
        device = hidden_state.device

        # ── 1. COMPUTE RAW INSTANTANEOUS SIGNALS ────────────────────────────────
        # These are the raw spikes before temporal smoothing

        # Stagnation
        discomfort_raw = self.path_detector(last_loss)

        # Identity / Harmony
        identity_buffer = hidden_state[:, :prefix_len, :]
        harmony_raw_tokens = F.cosine_similarity(
            identity_buffer, self.ego_vector.unsqueeze(1), dim=-1
        )  # (B, prefix_len)
        harmony_prefix  = harmony_raw_tokens * 2.0
        harmony_raw     = harmony_prefix.mean(dim=1)  # (B,) instantaneous

        # Curiosity
        curiosity_per_token = torch.zeros(B, T, device=device)
        if T > 1:
            current_states = hidden_state[:, :-1, :]
            target_states  = hidden_state[:, 1:, :].detach()
            predictions    = self.curiosity_predictor(current_states)
            error = F.mse_loss(predictions, target_states, reduction='none').mean(dim=-1)
            curiosity_per_token[:, 1:] = torch.tanh(error)
        curiosity_raw = curiosity_per_token.mean(dim=1)  # (B,) instantaneous

        # Pain
        pain_raw = self.survival.calculate_pain(mean_h, slow_grad_norm)  # (B,)

        # ── 2. TEMPORAL EGO DECAY — the emotional shadow system ──────────────
        # Each signal = decay * old_state + (1 - decay) * new_raw_spike
        # When raw signal is HIGH  → state rises toward new peak
        # When raw signal is LOW   → state decays gradually toward zero
        # The asymmetry: rising is instant, falling is slow — like real emotions

        pain_mean = pain_raw.mean().item()
        with torch.no_grad():
            # PAIN — rises fast on spike, decays slowly
            # If new pain > current state: jump up immediately (acute response)
            # If new pain < current state: decay gently (lingering trauma)
            new_pain = max(pain_mean, self.pain_state.item() * self.pain_decay)
            # But also blend in the new signal so recovery is possible
            new_pain = self.pain_decay * new_pain + (1 - self.pain_decay) * pain_mean
            self.pain_state.fill_(float(torch.clamp(torch.tensor(new_pain), 0.0, 3.0)))

            # DISCOMFORT — medium decay, fades when learning resumes
            disc_val = discomfort_raw.item() if hasattr(discomfort_raw, 'item') else float(discomfort_raw)
            new_disc = self.discomfort_decay * self.discomfort_state.item() \
                     + (1 - self.discomfort_decay) * disc_val
            self.discomfort_state.fill_(float(torch.clamp(torch.tensor(new_disc), 0.0, 1.0)))

            # CURIOSITY — fast decay, novelty is fleeting
            cur_mean = curiosity_raw.mean().item()
            new_cur  = self.curiosity_decay * self.curiosity_state.item() \
                     + (1 - self.curiosity_decay) * cur_mean
            self.curiosity_state.fill_(float(torch.clamp(torch.tensor(new_cur), 0.0, 2.0)))

            # HARMONY — slowest decay, trust builds and dissolves over time
            har_mean = harmony_raw.mean().item()
            new_har  = self.harmony_decay * self.harmony_state.item() \
                     + (1 - self.harmony_decay) * har_mean
            self.harmony_state.fill_(float(torch.clamp(torch.tensor(new_har), -2.0, 2.0)))

            # Track peaks — so downstream systems know if we're spiking or recovering
            self.pain_peak.fill_(max(pain_mean, self.pain_peak.item() * 0.99))
            self.harmony_peak.fill_(max(har_mean, self.harmony_peak.item() * 0.99))

        # ── 3. USE TEMPORALLY SMOOTHED SIGNALS ──────────────────────────────
        # These carry the emotional shadow forward
        pain_global      = self.pain_state.expand(B)          # (B,)
        discomfort_smooth = self.discomfort_state             # scalar
        curiosity_smooth  = self.curiosity_state              # scalar
        harmony_smooth    = self.harmony_state                # scalar

        # Rebuild per-token tensors using smoothed globals
        harmony_per_token = torch.zeros(B, T, device=device)
        harmony_per_token[:, :prefix_len] = harmony_prefix   # raw prefix alignment
        # Blend with smooth state so full sequence feels the temporal shadow
        harmony_per_token = harmony_per_token * 0.6 + harmony_smooth * 0.4

        # Curiosity per token — blend raw prediction error with smooth state
        curiosity_per_token = curiosity_per_token * 0.6 + curiosity_smooth * 0.4

        # ── 4. META-GATING ───────────────────────────────────────────────
        weights = self.meta_gate(mean_h)  # (B, 4)

        # Identity-gated curiosity — curiosity feels better when on-brand
        harmony_boost    = torch.sigmoid(torch.tensor(harmony_smooth * 5.0, device=device))
        curiosity_global = curiosity_per_token.mean(dim=1) * harmony_boost
        curiosity_per_token = curiosity_per_token * harmony_boost

        # ── 5. WILL SIGNAL ────────────────────────────────────────────────
        # Pain now has temporal weight — a bad moment echoes into future steps
        will_per_token = (
            (weights[:, 0].unsqueeze(1) + discomfort_smooth) * curiosity_per_token +
            (weights[:, 1].unsqueeze(1) * 2.0) * harmony_per_token -
            weights[:, 2].unsqueeze(1) * pain_global.unsqueeze(1)
        )
        will_global = will_per_token.mean(dim=1)

        # ── 6. RECOVERY SIGNAL — is pain declining? ──────────────────────
        # Positive = recovering, Negative = worsening
        # Downstream systems can use this to know if she's healing or spiraling
        pain_delta = pain_mean - self.pain_state.item()  # raw vs smoothed
        recovering = float(pain_delta < 0)               # 1.0 if improving

        return {
            "will_power"      : will_global,
            "harmony"         : torch.tensor(harmony_smooth, device=device).expand(B),
            "curiosity"       : curiosity_global,
            "pain"            : pain_global,
            "discomfort"      : discomfort_smooth,
            "will_per_token"  : will_per_token,
            "will_variance"   : will_per_token.var(dim=1),
            # New — temporal state for logging/consolidation
            "pain_peak"       : self.pain_peak.clone(),
            "harmony_peak"    : self.harmony_peak.clone(),
            "recovering"      : torch.tensor(recovering, device=device),
        }

    def reset_ego(self):
        """Call between sessions — let ego start fresh but not amnesiac.
        Partial reset: harmony carries over (trust persists), pain resets (new start).
        """
        self.pain_state.fill_(0.0)
        self.discomfort_state.fill_(0.0)
        self.curiosity_state.fill_(0.0)
        # Harmony deliberately NOT reset — trust persists across sessions
        self.pain_peak.fill_(0.0)
