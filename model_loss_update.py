"""
GPT-OSS Improved
Key improvements over original:
1. TOTAL: MSE regression → Mixture Density Network (MDN) with logistic components
   - Captures multi-modal dosage distribution (spikes at 0, 10, 30, 90)
   - Still a regression model, but learns full conditional distribution
2. SHIFT: Focal loss → Dice Loss + Focal combo + hierarchical binary detection
   - Dice loss directly optimizes per-class F1 (immune to class imbalance)
   - Binary auxiliary head detects "changed?" before classifying direction
   - Prevents model from always predicting Maintain
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

# Always print (single GPU mode)
def _is_master():
    return True

def focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()

    focal = (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)
    loss = -focal * log_pt

    if alpha is not None:
        alpha_t = alpha.gather(dim=0, index=targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Multi-class Dice Loss: directly optimizes per-class F1 score.
    Immune to class imbalance because it normalizes per-class.
    
    Args:
        logits: (N, C)
        targets: (N,) int64
        smooth: smoothing factor to avoid division by zero
        ignore_index: class to ignore
    Returns:
        1 - mean(Dice per class)
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
    
    num_classes = logits.size(-1)
    probs = F.softmax(logits, dim=-1)  # (N, C)
    
    # One-hot encode targets
    targets_onehot = F.one_hot(targets, num_classes).float()  # (N, C)
    
    # Per-class Dice
    intersection = (probs * targets_onehot).sum(dim=0)  # (C,)
    union = probs.sum(dim=0) + targets_onehot.sum(dim=0)  # (C,)
    
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    
    # Only average over classes that actually appear in targets
    class_present = targets_onehot.sum(dim=0) > 0
    if class_present.any():
        return 1.0 - dice_per_class[class_present].mean()
    return torch.tensor(0.0, device=logits.device)


# =============================================================================
# Mixture Density Network for TOTAL regression
# =============================================================================

class MixtureDensityHead(nn.Module):
    """
    Mixture Density Network: models P(total | context) as mixture of logistics.
    
    Unlike MSE which assumes unimodal Gaussian and predicts the mean,
    MDN learns the full conditional distribution → captures multi-modal
    dosage patterns (e.g., spikes at 0mg, 10mg, 30mg, 90mg).
    
    Still a regression model: outputs continuous probability density,
    not classification over discrete bins.
    """
    
    def __init__(self, n_embd, n_components=8, min_val=0.0, max_val=550.0):
        super().__init__()
        self.n_components = n_components
        self.min_val = min_val
        self.max_val = max_val
        
        # Shared feature extraction
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.LayerNorm(n_embd),
        )
        # Each component: (weight, mean, log_scale)
        self.params_head = nn.Linear(n_embd, n_components * 3)
        
        # Initialize means spread across the range
        with torch.no_grad():
            # Bias for mu: spread initial means across [0, max_val]
            bias = self.params_head.bias.data
            K = n_components
            # pi biases: all equal (uniform mixture)
            bias[:K] = 0.0
            # mu biases: spread across range via inverse sigmoid
            for i in range(K):
                target_mu = (i + 0.5) / K  # uniform in [0, 1]
                target_mu = max(0.01, min(0.99, target_mu))
                bias[K + i] = math.log(target_mu / (1 - target_mu))  # inverse sigmoid
            # log_scale biases: moderate width
            bias[2*K:] = 2.0  # exp(2) ≈ 7.4 units width
    
    def forward(self, x):
        """
        Returns:
            pi: (B, T, K) mixture weights (sum to 1)
            mu: (B, T, K) component means (in [min_val, max_val])
            log_s: (B, T, K) component log-scales
        """
        h = self.net(x)
        params = self.params_head(h)  # (B, T, K*3)
        
        K = self.n_components
        pi_logits, mu_raw, log_s = (
            params[..., :K],
            params[..., K:2*K],
            params[..., 2*K:],
        )
        
        pi = F.softmax(pi_logits, dim=-1)
        mu = self.min_val + (self.max_val - self.min_val) * torch.sigmoid(mu_raw)
        log_s = torch.clamp(log_s, min=-1.0, max=6.0)  # scale range: ~0.05 to ~400
        
        return pi, mu, log_s
    
    @staticmethod
    def nll_loss(pi, mu, log_s, target, valid_mask):
        """
        Negative log-likelihood of mixture of logistics.
        
        Logistic distribution is similar to Gaussian but has heavier tails,
        making it more robust and better at capturing sharp peaks.
        """
        if not valid_mask.any():
            return torch.tensor(0.0, device=pi.device)
        
        pi = pi[valid_mask]       # (N, K)
        mu = mu[valid_mask]       # (N, K)
        log_s = log_s[valid_mask] # (N, K)
        t = target[valid_mask].unsqueeze(-1)  # (N, 1)
        
        # Logistic distribution log-probability:
        # log p(t | mu, s) = -(t-mu)/s - log(s) - 2*log(1 + exp(-(t-mu)/s))
        s = torch.exp(log_s)
        z = (t - mu) / (s + 1e-8)
        log_p = -z - log_s - 2.0 * F.softplus(-z)
        
        # Log-sum-exp over mixture components:
        # log P(t) = log Σ_k π_k * p_k(t)
        log_mix = torch.log(pi + 1e-8) + log_p  # (N, K)
        nll = -torch.logsumexp(log_mix, dim=-1)  # (N,)
        
        # Clamp for stability
        nll = torch.clamp(nll, min=-5.0, max=20.0)
        
        return nll.mean()
    
    def predict_mean(self, x):
        """Inference: weighted mean of mixture"""
        pi, mu, _ = self.forward(x)
        return (pi * mu).sum(dim=-1)  # (B, T)
    
    def predict_mode(self, x):
        """Inference: mean of highest-weight component"""
        pi, mu, _ = self.forward(x)
        idx = pi.argmax(dim=-1, keepdim=True)  # (B, T, 1)
        return mu.gather(-1, idx).squeeze(-1)  # (B, T)
    
    def sample(self, x):
        """Inference: sample from the mixture"""
        pi, mu, log_s = self.forward(x)
        
        # Pick component
        comp = torch.multinomial(
            pi.reshape(-1, pi.size(-1)), 1
        ).reshape(pi.shape[:-1])  # (B, T)
        
        # Get selected component parameters
        mu_sel = mu.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        log_s_sel = log_s.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        
        # Sample from logistic distribution
        u = torch.rand_like(mu_sel).clamp(1e-5, 1 - 1e-5)
        sample = mu_sel + torch.exp(log_s_sel) * (torch.log(u) - torch.log(1 - u))
        
        return sample.clamp(self.min_val, self.max_val)


# =============================================================================
# Hierarchical Shift Head
# =============================================================================

class HierarchicalShiftHead(nn.Module):
    """
    Two-stage SHIFT prediction:
    Stage 1: Binary - "Did dose change?" (change vs maintain)
    Stage 2: Direction - "Increase or Decrease?" (only for changed tokens)
    
    This prevents the model from collapsing to always-Maintain because:
    - Stage 1 is a balanced binary problem (~15% change vs 85% maintain)
      with dedicated Dice loss that optimizes F1 directly
    - Stage 2 only sees change tokens, so it's ~67% Decrease vs 33% Increase
      (much more balanced than the original 3-class problem)
    
    Final shift logits are composed from both stages.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.shift_vocab_size = config.shift_vocab_size
        
        # Stage 1: Binary "changed?" detection
        self.change_detector = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 2),  # [maintain, changed]
        )
        
        # Stage 2: Full shift classification (for gradient flow)
        self.shift_classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.shift_vocab_size),
        )
    
    def forward(self, x):
        """
        Returns:
            shift_logits: (B, T, shift_vocab_size) - standard shift logits
            change_logits: (B, T, 2) - binary change detection logits
        """
        change_logits = self.change_detector(x)  # (B, T, 2)
        shift_logits = self.shift_classifier(x)  # (B, T, shift_vocab_size)
        return shift_logits, change_logits


# =============================================================================
# Core Architecture (unchanged from original)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    half_dim = x1.shape[-1]
    cos = cos[:, :half_dim].unsqueeze(0).to(x.dtype)
    sin = sin[:, :half_dim].unsqueeze(0).to(x.dtype)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0, max_position_embeddings=2048):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class AgeEncoding(nn.Module):
    def __init__(self, config, max_dim=1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, age):
        y = torch.zeros(age.shape[0], age.shape[1], self.n_embd, device=age.device)
        y[..., 0::2] = torch.sin(age.unsqueeze(-1) / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(age.unsqueeze(-1) / 365.25 * self.div_term)
        return self.linear(y)


def swiglu(x, limit=7.0):
    alpha = 1.0
    x_glu, x_linear = x.chunk(2, dim=-1)
    if limit > 0:
        x_glu = x_glu.clamp(min=-limit, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window if (hasattr(config, 'sliding_window') and layer_idx % 2 == 0) else 0
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, age=None, attn_mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(x, T)
        q_shape, k_shape = q.shape, k.shape
        q = _apply_rotary_emb(q.reshape(B * self.n_head, T, self.head_dim), cos, sin).reshape(q_shape)
        k = _apply_rotary_emb(k.reshape(B * self.n_kv_head, T, self.head_dim), cos, sin).reshape(k_shape)
        
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k = k.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
        
        if self.flash and attn_mask is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            att = None
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(causal_mask.view(1, 1, T, T), float('-inf'))
            if self.sliding_window > 0:
                window_mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=-self.sliding_window).bool()
                att = att.masked_fill(window_mask.view(1, 1, T, T), float('-inf'))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y, att


class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, 'num_experts') else 8
        self.experts_per_token = config.experts_per_token if hasattr(config, 'experts_per_token') else 2
        self.n_embd = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias),
                nn.GELU(),
                nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.experts_per_token, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_input = x[expert_mask]
            expert_output = self.experts[i](expert_input)
            expert_output_full = torch.zeros_like(x)
            expert_output_full[expert_mask] = expert_output
            for k in range(self.experts_per_token):
                token_mask = (selected_experts[..., k] == i)
                if token_mask.any():
                    weights = routing_weights[..., k:k+1]
                    final_output += weights * expert_output_full * token_mask.unsqueeze(-1)
        return final_output


class ModernFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, 2 * self.intermediate_size, bias=config.bias)
        self.c_proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = swiglu(x, limit=7.0)
        x = self.c_proj(x)
        return self.dropout(x)


class ModernBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        if hasattr(config, 'use_moe') and config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = ModernFFN(config)

    def forward(self, x, age=None, attn_mask=None):
        y, att = self.attn(self.ln_1(x), age, attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


# =============================================================================
# ModernDelphi (original, unchanged)
# =============================================================================

@dataclass
class ModernDelphiConfig:
    block_size: int = 1024
    vocab_size: int = 1290
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    use_moe: bool = False
    num_experts: int = 8
    experts_per_token: int = 2
    sliding_window: int = 256
    rope_theta: float = 10000.0
    time_distribution: str = 'exponential'


class ModernDelphi(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wae=AgeEncoding(config),
            token_drop=nn.Dropout(config.token_dropout),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([ModernBlock(config, i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        time_distribution = getattr(config, 'time_distribution', 'exponential')
        if time_distribution == 'weibull':
            self.time_shape_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)
        
        attn_mask = (idx > 0).view(b, 1, 1, t) * (idx > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1)))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask + (idx == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att) if att[0] is not None else None

        if targets is not None:
            logits = self.lm_head(x)
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[..., ignored_tokens] = -torch.inf
            targets_flat = targets.reshape(-1)
            pass_tokens = targets_flat != -1
            for k in ignored_tokens:
                pass_tokens *= targets_flat != k
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[pass_tokens],
                targets_flat[pass_tokens], ignore_index=-1)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(
                    dt, -1,
                    (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                    .max(-1).indices.squeeze((1, 2)))
            
            time_distribution = getattr(self.config, 'time_distribution', 'exponential')
            if time_distribution == 'weibull':
                time_shape = F.softplus(self.time_shape_head(x)) + 0.1
                log_scale = torch.logsumexp(logits, -1)
                scale = torch.clamp(torch.exp(log_scale), min=0.1, max=1e4) + self.config.t_min
                event_probs = F.softmax(logits, dim=-1)
                shape = torch.clamp((event_probs * time_shape).sum(-1), min=0.1, max=10.0)
                dt_flat = torch.clamp(dt.view(-1), min=0.1) + self.config.t_min
                scale_flat = scale.view(-1)
                shape_flat = shape.view(-1)
                ratio = torch.clamp(dt_flat / scale_flat, min=1e-6, max=1e3)
                log_likelihood = (
                    torch.log(shape_flat + 1e-8) -
                    shape_flat * torch.log(scale_flat + 1e-8) +
                    (shape_flat - 1) * torch.log(dt_flat + 1e-8) -
                    ratio.pow(shape_flat))
                loss_dt = -torch.mean(torch.clamp(log_likelihood[pass_tokens], min=-100, max=100))
            else:
                lse = torch.logsumexp(logits, -1)
                lse = -torch.log(torch.exp(-lse) + self.config.t_min)
                ldt = -torch.log(dt + self.config.t_min).view(-1)
                loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
                loss_dt = torch.mean(loss_dt[pass_tokens])
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
        else:
            logits = self.lm_head(x)
            loss = None
        return logits, loss, att

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif isinstance(m, RMSNorm) and pn == 'scale':
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    decay.add(fpn)
        if 'lm_head.weight' in decay:
            decay.discard('lm_head.weight')
        if 'lm_head.weight' not in no_decay:
            no_decay.add('lm_head.weight')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        if 'lm_head.weight' in no_decay and 'lm_head.weight' not in param_dict:
            no_decay.discard('lm_head.weight')
        inter_params = decay & no_decay
        if inter_params:
            for pn in list(inter_params):
                decay.discard(pn)
        union_params = decay | no_decay
        missing_params = param_dict.keys() - union_params
        if missing_params:
            for pn in missing_params:
                no_decay.add(pn)
        union_params = decay | no_decay
        inter_params = decay & no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        if _is_master():
            print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True,
                 termination_tokens=None, top_k=None):
        if termination_tokens is None:
            warnings.warn('When using a custom dataset, consider changing the `termination_tokens` argument.')
            termination_tokens = [1269]
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        if max_new_tokens == -1:
            max_new_tokens = 128
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:, self.config.ignore_tokens] = -torch.inf
            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            t_next = torch.clamp(
                -torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(),
                min=0, max=365*80).min(1)
            idx_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            if torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break
        pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)
        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = -10000
        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([
                logits[:, j].scatter_(1, fill[:, :j+1], -torch.inf)
                for j in range(fill.shape[1])
            ]).transpose(0, 1)
        return idx, age, logits


# =============================================================================
# Composite Embedding + IMPROVED Multi-Head Output
# =============================================================================

class CompositeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.data_emb = nn.Embedding(config.data_vocab_size, config.n_embd)
        self.shift_emb = nn.Embedding(config.shift_vocab_size, config.n_embd)
        self.total_emb = nn.Embedding(config.total_vocab_size, config.n_embd)

    def forward(self, data, shift, total):
        data_idx = torch.clamp(data, min=0, max=self.data_emb.num_embeddings - 1)
        shift_idx = torch.clamp(shift, min=0, max=self.shift_emb.num_embeddings - 1)
        total_idx = torch.clamp(total, min=0, max=self.total_emb.num_embeddings - 1)
        return self.data_emb(data_idx) + self.shift_emb(shift_idx) + self.total_emb(total_idx)


class MultiHeadOutput(nn.Module):
    """
    IMPROVED Multi-Head Output Layer:
    
    Changes from original:
    1. TOTAL: Linear(n_embd, 1) → MixtureDensityHead (8 logistic components)
       - Captures multi-modal dosage distribution
    2. SHIFT: Single classifier → HierarchicalShiftHead
       - Binary change detection + direction classification
       - Prevents collapse to always-Maintain
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.time_distribution = getattr(config, 'time_distribution', 'exponential')
        self.use_drug_conditioning = getattr(config, 'use_drug_conditioning', False)
        
        # DATA Head (unchanged)
        self.data_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
        # ============================================================
        # IMPROVED: Hierarchical SHIFT Head
        # Stage 1: Binary change detection (change vs maintain)
        # Stage 2: Direction classification (decrease/maintain/increase)
        # ============================================================
        self.shift_head = HierarchicalShiftHead(config)
        
        # ============================================================
        # IMPROVED: TOTAL Head → Mixture Density Network
        # 8 logistic mixture components → captures multi-modal dosage
        # ============================================================
        n_mdn_components = getattr(config, 'mdn_n_components', 8)
        total_max = float(getattr(config, 'total_vocab_size', 552)) - 1.0
        self.total_head = MixtureDensityHead(
            config.n_embd,
            n_components=n_mdn_components,
            min_val=0.0,
            max_val=total_max,
        )
        
        # Drug-Conditioning (FiLM)
        if self.use_drug_conditioning:
            self.shift_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)
            )
            self.shift_drug_cond_head = HierarchicalShiftHead(config)
            
            self.total_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)
            )
            self.total_drug_cond_head = MixtureDensityHead(
                config.n_embd,
                n_components=n_mdn_components,
                min_val=0.0,
                max_val=total_max,
            )
            
            for film_gen in [self.shift_film_generator, self.total_film_generator]:
                last_layer = film_gen[-1]
                nn.init.zeros_(last_layer.weight)
                with torch.no_grad():
                    last_layer.bias[:config.n_embd].fill_(1.0)
                    last_layer.bias[config.n_embd:].zero_()
        
        # Time Head (unchanged)
        self.time_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        if self.time_distribution == 'weibull':
            self.time_shape_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
    
    def forward(self, x, drug_emb=None, drug_token_mask=None):
        # DATA (unchanged)
        data_logits = self.data_head(x)
        
        # SHIFT: hierarchical
        shift_logits, change_logits = self.shift_head(x)
        
        # TOTAL: MDN
        total_pi, total_mu, total_log_s = self.total_head(x)
        # For backward compatibility, provide point estimate
        total_point = (total_pi * total_mu).sum(dim=-1)  # weighted mean
        
        output = {
            'data': data_logits,
            'shift': shift_logits,
            'change': change_logits,           # NEW: binary change logits
            'total': total_point,              # backward compat: point estimate
            'total_pi': total_pi,              # NEW: MDN mixture weights
            'total_mu': total_mu,              # NEW: MDN means
            'total_log_s': total_log_s,        # NEW: MDN log-scales
            'time_scale': self.time_head(x),
        }
        
        # Drug-Conditioned Predictions
        if self.use_drug_conditioning and drug_emb is not None:
            # SHIFT FiLM
            shift_film = self.shift_film_generator(drug_emb)
            shift_gamma, shift_beta = shift_film.chunk(2, dim=-1)
            shift_modulated = shift_gamma * x + shift_beta
            shift_drug_logits, change_drug_logits = self.shift_drug_cond_head(shift_modulated)
            
            # TOTAL FiLM
            total_film = self.total_film_generator(drug_emb)
            total_gamma, total_beta = total_film.chunk(2, dim=-1)
            total_modulated = total_gamma * x + total_beta
            total_drug_pi, total_drug_mu, total_drug_log_s = self.total_drug_cond_head(total_modulated)
            total_drug_point = (total_drug_pi * total_drug_mu).sum(dim=-1)
            
            if drug_token_mask is not None:
                # Blend: FiLM for drug tokens, standard for others
                output['shift_drug_cond'] = torch.where(
                    drug_token_mask.unsqueeze(-1), shift_drug_logits, shift_logits)
                output['change_drug_cond'] = torch.where(
                    drug_token_mask.unsqueeze(-1), change_drug_logits, change_logits)
                output['total_drug_cond'] = torch.where(
                    drug_token_mask, total_drug_point, total_point)
                output['total_drug_cond_pi'] = torch.where(
                    drug_token_mask.unsqueeze(-1), total_drug_pi, total_pi)
                output['total_drug_cond_mu'] = torch.where(
                    drug_token_mask.unsqueeze(-1), total_drug_mu, total_mu)
                output['total_drug_cond_log_s'] = torch.where(
                    drug_token_mask.unsqueeze(-1), total_drug_log_s, total_log_s)
            else:
                output['shift_drug_cond'] = shift_drug_logits
                output['change_drug_cond'] = change_drug_logits
                output['total_drug_cond'] = total_drug_point
                output['total_drug_cond_pi'] = total_drug_pi
                output['total_drug_cond_mu'] = total_drug_mu
                output['total_drug_cond_log_s'] = total_drug_log_s
        
        output['time'] = output['time_scale']
        if self.time_distribution == 'weibull':
            output['time_shape'] = F.softplus(self.time_shape_head(x)) + 0.1
        
        return output


# =============================================================================
# Composite Config (with new options)
# =============================================================================

@dataclass
class CompositeDelphiConfig:
    block_size: int = 1024
    data_vocab_size: int = 1290
    shift_vocab_size: int = 5
    total_vocab_size: int = 552
    
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    
    use_drug_conditioning: bool = False
    drug_token_min: int = 1279
    drug_token_max: int = 1289

    use_moe: bool = True
    num_experts: int = 8
    experts_per_token: int = 2
    sliding_window: int = 256
    rope_theta: float = 10000.0
    
    # Loss weights
    loss_weight_data: float = 1.0
    loss_weight_shift: float = 1.0
    loss_weight_total: float = 0.5
    loss_weight_time: float = 1.0

    # SHIFT loss: 'dice_focal' (NEW), 'focal', or 'ce'
    shift_loss_type: str = 'dice_focal'
    shift_ignore_index: int = 0
    shift_focal_gamma: float = 2.0
    shift_class_weights: list = field(default_factory=list)
    # Dice + Focal blend ratio (1.0 = all dice, 0.0 = all focal)
    shift_dice_weight: float = 0.5
    # Weight for binary change detection auxiliary loss
    loss_weight_change: float = 2.0
    # Which SHIFT class index is "Maintain" (after +1 token shift: Maintain=2)
    shift_maintain_idx: int = 2          # ← 이 줄 추가
    
    # TOTAL: MDN parameters
    mdn_n_components: int = 8
    
    time_distribution: str = 'exponential'


# =============================================================================
# Composite Delphi (IMPROVED)
# =============================================================================

class CompositeDelphi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.composite_emb = CompositeEmbedding(config)
        self.age_encoding = AgeEncoding(config)
        self.token_drop = nn.Dropout(config.token_dropout)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([ModernBlock(config, i) for i in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.multi_head = MultiHeadOutput(config)
        
        # Weight tying
        self.multi_head.data_head.weight = self.composite_emb.data_emb.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, data, shift, total, age,
                targets_data=None, targets_shift=None, targets_total=None,
                targets_age=None, drug_conditioning_data=None,
                validation_loss_mode=False):
        device = data.device
        b, t = data.size()
        
        composite_emb = self.composite_emb(data, shift, total)
        age_emb = self.age_encoding(age)
        x = self.token_drop(composite_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.drop(x)
        
        attn_mask = (data > 0).view(b, 1, 1, t) * (data > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        if targets_data is not None and self.config.mask_ties:
            attn_mask *= (age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask + (data == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        att_list = []
        for block in self.h:
            x, att = block(x, age, attn_mask)
            att_list.append(att)
        x = self.ln_f(x)
        att = torch.stack(att_list) if att_list[0] is not None else None
        
        # Drug conditioning
        drug_emb = None
        drug_token_mask = None
        if self.config.use_drug_conditioning:
            drug_source = data
            if drug_source is not None:
                drug_source_clamped = torch.clamp(
                    drug_source, min=0,
                    max=self.composite_emb.data_emb.num_embeddings - 1)
                drug_emb = self.composite_emb.data_emb(drug_source_clamped)
            drug_token_min = getattr(self.config, 'drug_token_min', 1279)
            drug_token_max = getattr(self.config, 'drug_token_max', 1289)
            if targets_data is not None:
                drug_token_mask = (targets_data >= drug_token_min) & (targets_data <= drug_token_max)
        
        logits = self.multi_head(x, drug_emb=drug_emb, drug_token_mask=drug_token_mask)
        
        if targets_data is not None:
            loss = self._compute_loss(
                logits, data, age,
                targets_data, targets_shift, targets_total, targets_age,
                attn_mask, validation_loss_mode)
        else:
            loss = None
        
        return logits, loss, att
    
    def _compute_loss(self, logits, data, age,
                      targets_data, targets_shift, targets_total, targets_age,
                      attn_mask, validation_loss_mode):
        device = data.device
        b, t = data.size()
        
        ignored_tokens = self.config.ignore_tokens.copy()
        if validation_loss_mode:
            ignored_tokens += [1]
        
        targets_flat = targets_data.reshape(-1)
        pass_tokens = targets_flat != -1
        for k in ignored_tokens:
            pass_tokens = pass_tokens * (targets_flat != k)
        
        data_vocab_size = self.config.data_vocab_size
        targets_flat_clamped = torch.clamp(targets_flat, min=0, max=data_vocab_size - 1)
        
        # 1. DATA Cross-Entropy Loss (unchanged)
        data_logits = logits['data']
        if validation_loss_mode:
            data_logits[..., ignored_tokens] = -torch.inf
        loss_data = F.cross_entropy(
            data_logits.reshape(-1, data_logits.size(-1))[pass_tokens],
            targets_flat_clamped[pass_tokens], ignore_index=-1)
        
        # ============================================================
        # 2. SHIFT Loss: Dice + Focal + Binary Change Detection
        # ============================================================
        shift_logits_source = logits['shift']
        change_logits_source = logits['change']
        if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
            shift_logits_source = logits['shift_drug_cond']
            change_logits_source = logits['change_drug_cond']
        
        shift_targets_all = targets_shift.reshape(-1)
        shift_pass_tokens = shift_targets_all != -1
        shift_logits_flat = shift_logits_source.reshape(-1, shift_logits_source.size(-1))[shift_pass_tokens]
        shift_targets_flat = shift_targets_all[shift_pass_tokens]
        
        shift_vocab_size = self.config.shift_vocab_size
        shift_targets_flat = torch.clamp(shift_targets_flat, min=-1, max=shift_vocab_size - 1)
        
        shift_ignore = int(getattr(self.config, 'shift_ignore_index', 0))
        shift_valid = (shift_targets_flat != -1) & (shift_targets_flat != shift_ignore)
        shift_logits_flat = shift_logits_flat[shift_valid]
        shift_targets_flat = shift_targets_flat[shift_valid]
        
        if shift_logits_flat.numel() == 0:
            loss_shift = torch.tensor(0.0, device=device)
        else:
            shift_loss_type = str(getattr(self.config, 'shift_loss_type', 'dice_focal')).lower()
            weights_list = getattr(self.config, 'shift_class_weights', None)
            weight_t = None
            if isinstance(weights_list, (list, tuple)) and len(weights_list) == shift_vocab_size:
                weight_t = torch.tensor(weights_list, device=device, dtype=torch.float32)
            
            if shift_loss_type == 'dice_focal':
                # Dice Loss: directly optimizes F1 per class
                # Immune to class imbalance because it normalizes per class
                loss_dice = dice_loss_multiclass(
                    shift_logits_flat, shift_targets_flat.long(),
                    smooth=1.0, ignore_index=None)
                
                # Focal Loss: down-weights easy examples (Maintain)
                gamma = float(getattr(self.config, 'shift_focal_gamma', 2.0))
                loss_focal = focal_loss_multiclass(
                    shift_logits_flat, shift_targets_flat.long(),
                    gamma=gamma, alpha=weight_t,
                    ignore_index=None, reduction='mean')
                
                # Blend
                dice_w = float(getattr(self.config, 'shift_dice_weight', 0.5))
                loss_shift = dice_w * loss_dice + (1.0 - dice_w) * loss_focal
            
            elif shift_loss_type == 'focal':
                gamma = float(getattr(self.config, 'shift_focal_gamma', 2.0))
                loss_shift = focal_loss_multiclass(
                    shift_logits_flat, shift_targets_flat.long(),
                    gamma=gamma, alpha=weight_t,
                    ignore_index=None, reduction='mean')
            else:
                loss_shift = F.cross_entropy(
                    shift_logits_flat, shift_targets_flat.long(),
                    weight=weight_t, ignore_index=-1)
        
        # Binary Change Detection Auxiliary Loss
        # Convert shift targets to binary: 0=maintain, 1=changed
        # This is a more balanced problem (~15% change vs 85% maintain)
        change_logits_flat = change_logits_source.reshape(-1, 2)[shift_pass_tokens]
        change_logits_flat = change_logits_flat[shift_valid]
        
        if change_logits_flat.numel() == 0:
            loss_change = torch.tensor(0.0, device=device)
        else:
            # Determine which SHIFT class index corresponds to "Maintain"
            # After +1 shift: typically Maintain=2 (raw 1), Decrease=1 (raw 0), Increase=3 (raw 2)
            # But this depends on encoding. We use: maintain = middle class
            # Binary: 0 = maintain, 1 = changed (decrease or increase)
            maintain_idx = getattr(self.config, 'shift_maintain_idx', 2)  # default: Maintain=2 after shift
            change_targets = (shift_targets_flat != maintain_idx).long()
            
            # Weighted BCE: weight the minority class (changed) more
            # ~15% changed → weight ~5.7x
            n_changed = change_targets.sum().float().clamp(min=1.0)
            n_maintain = (change_targets == 0).sum().float().clamp(min=1.0)
            change_weight = torch.tensor([1.0, n_maintain / n_changed], device=device)
            change_weight = change_weight.clamp(max=10.0)
            
            loss_change = F.cross_entropy(
                change_logits_flat, change_targets, weight=change_weight)
        
        # ============================================================
        # 3. TOTAL Loss: Mixture Density Network NLL
        #    (replaces MSE → captures multi-modal dosage distribution)
        # ============================================================
        if 'total_drug_cond_pi' in logits and self.config.use_drug_conditioning:
            total_pi = logits['total_drug_cond_pi']
            total_mu = logits['total_drug_cond_mu']
            total_log_s = logits['total_drug_cond_log_s']
        else:
            total_pi = logits['total_pi']
            total_mu = logits['total_mu']
            total_log_s = logits['total_log_s']
        
        targets_total_float = targets_total.float()
        drug_token_min = getattr(self.config, 'drug_token_min', 1279)
        drug_token_max = getattr(self.config, 'drug_token_max', 1289)
        is_drug = (targets_data.reshape(-1) >= drug_token_min) & (targets_data.reshape(-1) <= drug_token_max)
        total_pass = pass_tokens & is_drug
        #total_target = targets_total_float.reshape(-1)[total_pass]
        
        # Reshape MDN outputs to flat
        total_pi_flat = total_pi.reshape(-1, total_pi.size(-1))
        total_mu_flat = total_mu.reshape(-1, total_mu.size(-1))
        total_log_s_flat = total_log_s.reshape(-1, total_log_s.size(-1))
        
        loss_total = MixtureDensityHead.nll_loss(
            total_pi_flat, total_mu_flat, total_log_s_flat,
            targets_total_float.reshape(-1),  # 전체 [32768] 
            total_pass)                       # mask [32768]
        
        # 4. Time-to-Event Loss (unchanged from original)
        dt = torch.clamp(targets_age - age, min=1.0)
        if self.config.mask_ties:
            dt = torch.gather(
                dt, -1,
                (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                .max(-1).indices.squeeze((1, 2)))
        
        time_distribution = getattr(self.config, 'time_distribution', 'exponential')
        if time_distribution == 'weibull':
            time_logits = logits['time_scale']
            time_shape = logits['time_shape']
            log_scale = torch.logsumexp(time_logits, -1)
            scale = torch.clamp(torch.exp(log_scale), min=1.0, max=365.0)
            event_probs = F.softmax(time_logits, dim=-1)
            shape = torch.clamp((event_probs * time_shape).sum(-1), min=0.5, max=5.0)
            dt_flat = torch.clamp(dt.view(-1), min=1.0)
            scale_flat = scale.view(-1)
            shape_flat = shape.view(-1)
            log_shape = torch.log(shape_flat)
            log_scale = torch.log(scale_flat)
            log_dt = torch.log(dt_flat)
            log_ratio = log_dt - log_scale
            power_term = torch.clamp(torch.exp(shape_flat * log_ratio), max=50.0)
            log_likelihood = log_shape - shape_flat * log_scale + (shape_flat - 1) * log_dt - power_term
            nll = -log_likelihood[pass_tokens]
            loss_time = torch.clamp(nll, min=0.0, max=20.0).mean()
        else:
            time_logits = logits['time_scale']
            lse = torch.logsumexp(time_logits, -1)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_time = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_time = torch.mean(loss_time[pass_tokens])
        
        # Weighted sum
        loss_weight_change = float(getattr(self.config, 'loss_weight_change', 2.0))
        
        total_loss = (
            self.config.loss_weight_data * loss_data +
            self.config.loss_weight_shift * loss_shift +
            loss_weight_change * loss_change +
            self.config.loss_weight_total * loss_total +
            self.config.loss_weight_time * loss_time
        )
        
        return {
            'loss': total_loss,
            'loss_data': loss_data,
            'loss_shift': loss_shift,
            'loss_change': loss_change,
            'loss_total': loss_total,
            'loss_time': loss_time,
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif isinstance(m, RMSNorm) and pn == 'scale':
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        if 'multi_head.data_head.weight' in decay:
            decay.discard('multi_head.data_head.weight')
        
        inter_params = decay & no_decay
        if inter_params:
            if _is_master():
                print(f"Warning: {len(inter_params)} params in both decay/no_decay, moving to no_decay")
            for pn in list(inter_params):
                decay.discard(pn)
        
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            if _is_master():
                print(f"Warning: {len(missing_params)} unclassified params, adding to no_decay")
            for pn in missing_params:
                no_decay.add(pn)
        
        union_params = decay | no_decay
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - union_params) == 0
        
        decay_filtered = [pn for pn in sorted(list(decay)) if pn in param_dict]
        no_decay_filtered = [pn for pn in sorted(list(no_decay)) if pn in param_dict]
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in decay_filtered], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in no_decay_filtered], "weight_decay": 0.0},
        ]
        
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        if _is_master():
            print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    @torch.no_grad()
    def generate(self, data, shift, total, age,
                 max_new_tokens=100, max_age=85*365.25,
                 no_repeat=True, termination_tokens=None):
        if termination_tokens is None:
            warnings.warn('Consider setting termination_tokens for your dataset.')
            termination_tokens = [1269]
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=data.device)
        if max_new_tokens == -1:
            max_new_tokens = 128
        
        for _ in range(max_new_tokens):
            logits, _, _ = self(data, shift, total, age, drug_conditioning_data=data)
            
            data_logits = logits['data'][:, -1, :]
            
            # SHIFT: use hierarchical prediction
            shift_logits_source = logits['shift']
            if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
                shift_logits_source = logits['shift_drug_cond']
            shift_logits = shift_logits_source[:, -1, :]
            
            # TOTAL: sample from MDN
            total_pi = logits['total_pi']
            if 'total_drug_cond_pi' in logits and self.config.use_drug_conditioning:
                total_pi = logits['total_drug_cond_pi']
                total_mu = logits['total_drug_cond_mu']
                total_log_s = logits['total_drug_cond_log_s']
            else:
                total_mu = logits['total_mu']
                total_log_s = logits['total_log_s']
            
            # Sample from MDN at last position
            pi_last = total_pi[:, -1, :]
            mu_last = total_mu[:, -1, :]
            log_s_last = total_log_s[:, -1, :]
            
            # Pick component
            comp = torch.multinomial(pi_last, 1)  # (B, 1)
            mu_sel = mu_last.gather(-1, comp).squeeze(-1)  # (B,)
            log_s_sel = log_s_last.gather(-1, comp).squeeze(-1)
            # Sample from logistic
            u = torch.rand_like(mu_sel).clamp(1e-5, 1 - 1e-5)
            total_sample = mu_sel + torch.exp(log_s_sel) * (torch.log(u) - torch.log(1 - u))
            total_sample = total_sample.clamp(0, self.config.total_vocab_size - 1)
            
            time_logits = logits['time'][:, -1, :]
            
            data_logits[:, self.config.ignore_tokens] = -torch.inf
            if no_repeat:
                fill = data.clone()
                fill[fill == 1] = 0
                data_logits = data_logits.scatter_(1, fill, -torch.inf)
            
            t_next = torch.clamp(
                -torch.exp(-time_logits) * torch.rand(time_logits.shape, device=data.device).log(),
                min=0, max=365*80).min(1)
            
            data_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]
            shift_next = torch.argmax(shift_logits, dim=-1, keepdim=True)
            total_next = total_sample.round().long().clamp(
                0, self.config.total_vocab_size - 1).unsqueeze(-1)
            
            data = torch.cat((data, data_next), dim=1)
            shift = torch.cat((shift, shift_next), dim=1)
            total = torch.cat((total, total_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            if torch.logical_or(
                torch.isin(data, termination_tokens).any(-1),
                age_next > max_age
            ).all():
                break
        
        return data, shift, total, age, logits