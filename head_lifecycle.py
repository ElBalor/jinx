"""
Head Lifecycle Manager for Guided Civilization Architecture.
Tracks per-head metrics (entropy, reward correlation) and implements:
1. Promotion (Reward Connection)
2. Demotion (Pruning Connection)
3. Re-assignment (Cell Repurposing / Reincarnation)
"""
import torch
import random
from typing import Dict, Optional, List

class HeadMetrics:
    """Tracks metrics for a single attention head."""
    def __init__(self, head_idx: int, bias_type: str):
        self.head_idx = head_idx
        self.bias_type = bias_type
        
        # Running statistics (EMA)
        self.entropy_ema = 2.0  # Start neutral
        self.reward_corr_ema = 0.0
        self.activation_freq = 0.5
        
        # Career History
        self.promotion_count = 0
        self.demotion_count = 0
        self.age = 0  # Steps since last re-assignment
        self.last_promotion_step = -1
        self.last_demotion_step = -1
    
    def update(self, entropy: float, reward_signal: float, active: bool):
        alpha = 0.05 # Fast adaptation for experiments
        self.entropy_ema = (1 - alpha) * self.entropy_ema + alpha * entropy
        
        if active:
            # Simple reward correlation proxy
            self.reward_corr_ema = (1 - alpha) * self.reward_corr_ema + alpha * reward_signal
            self.activation_freq = (1 - alpha) * self.activation_freq + alpha * 1.0
        else:
            self.activation_freq = (1 - alpha) * self.activation_freq
            
        self.age += 1

class HeadLifecycleManager:
    def __init__(self, n_heads: int, bias_types: List[str], promotion_threshold: float = 0.7, demotion_threshold: float = 0.2):
        self.n_heads = n_heads
        self.current_roles = bias_types.copy() 
        self.metrics = [HeadMetrics(i, t) for i, t in enumerate(bias_types)]
        
        # Civilization Rules
        self.promotion_thresh = promotion_threshold
        self.demotion_thresh = demotion_threshold
        self.max_demotions = 3       # "Three strikes" logic
        
        # Re-assignment Pool (The "Stem Cells")
        self.role_pool = ['local', 'global', 'causal', 'random', 'instruction', 'order']

    def update_metrics(self, head_idx: int, attention_probs: torch.Tensor, reward_signal: float = 0.0):
        if head_idx >= len(self.metrics):
            return
        
        # Compute entropy
        p = attention_probs.clamp_min(1e-8)
        entropy = -(p * p.log()).sum().item()
        
        # active if peak attention is high
        active = attention_probs.max().item() > 0.1
        self.metrics[head_idx].update(entropy, reward_signal, active)

    def evaluate_promotion(self, head_idx: int, step: int) -> bool:
        m = self.metrics[head_idx]
        if step - m.last_promotion_step < 100: return False
        
        if m.reward_corr_ema > self.promotion_thresh and m.activation_freq > 0.3:
            m.promotion_count += 1
            m.last_promotion_step = step
            m.reward_corr_ema *= 0.8 # Decay to require sustained proof
            return True
        return False

    def evaluate_demotion(self, head_idx: int, step: int) -> bool:
        m = self.metrics[head_idx]
        if step - m.last_demotion_step < 100: return False
        
        if m.reward_corr_ema < self.demotion_thresh or m.activation_freq < 0.1:
            m.demotion_count += 1
            m.last_demotion_step = step
            return True
        return False

    def reassign_role(self, head_idx: int) -> Optional[str]:
        m = self.metrics[head_idx]
        if m.demotion_count >= self.max_demotions:
            new_role = random.choice(self.role_pool)
            self.current_roles[head_idx] = new_role
            m.bias_type = new_role
            m.promotion_count = 0
            m.demotion_count = 0
            m.age = 0
            m.entropy_ema = 2.0
            m.reward_corr_ema = 0.0
            return new_role
        return None

    def get_promotion_gain_multiplier(self, head_idx: int) -> float:
        """Get gain multiplier for promoted heads (increase plasticity)."""
        m = self.metrics[head_idx]
        return 1.0 + 0.2 * m.promotion_count
    
    def get_demotion_gain_multiplier(self, head_idx: int) -> float:
        """Get gain multiplier for demoted heads (decrease plasticity)."""
        m = self.metrics[head_idx]
        return max(0.1, 1.0 - 0.2 * m.demotion_count)

    def get_plasticity_multipliers(self):
        """Returns per-head plasticity scaling factor based on career status."""
        mults = []
        for m in self.metrics:
            val = 1.0 + (m.promotion_count * 0.2) - (m.demotion_count * 0.2)
            mults.append(max(0.1, min(3.0, val)))
        return torch.tensor(mults)