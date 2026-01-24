"""
GPT-OSS
- MoE (Mixture of Experts) for domain-specific learning
- Sliding Window Attention for long medical histories
- RoPE (Rotary Position Embedding) 
- AgeEncoding for temporal medical events
- Custom medical loss functions
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
    """
    Multi-class focal loss.

    Args:
        logits: (N, C)
        targets: (N,) int64
        gamma: focusing parameter
        alpha: optional per-class weights (C,)
        ignore_index: optional class id to ignore
        reduction: 'mean' | 'sum' | 'none'
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (N,)
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""
    
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


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding
    
    Args:
        x: (B * n_head, T, head_dim)
        cos, sin: (T, head_dim)
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)  # 각각 (B * n_head, T, head_dim // 2)
    
    # cos, sin을 (1, T, head_dim // 2)로 변환하여 broadcasting 가능하게 함
    half_dim = x1.shape[-1]
    cos = cos[:, :half_dim].unsqueeze(0).to(x.dtype)  # (1, T, head_dim // 2)
    sin = sin[:, :half_dim].unsqueeze(0).to(x.dtype)  # (1, T, head_dim // 2)
    
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE with medical age information"""
    
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Args:
            x: input tensor
            seq_len: sequence length
        Returns:
            cos, sin tensors for rotary embedding
        """
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class AgeEncoding(nn.Module):
    """Delphi's signature age encoding for medical events"""
    
    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, age):
        """
        Args:
            age: age tensor in days, shape (B, T)
        Returns:
            age embeddings, shape (B, T, n_embd)
        """
        y = torch.zeros(age.shape[0], age.shape[1], self.n_embd, device=age.device)
        y[..., 0::2] = torch.sin(age.unsqueeze(-1) / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(age.unsqueeze(-1) / 365.25 * self.div_term)
        y = self.linear(y)
        return y


def swiglu(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """SwiGLU activation with optional limiting"""
    alpha = 1.0
    x_glu, x_linear = x.chunk(2, dim=-1)
    if limit > 0:
        x_glu = x_glu.clamp(min=-limit, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with sliding window support"""
    
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        
        # Sliding window: apply to every other layer
        self.sliding_window = config.sliding_window if (hasattr(config, 'sliding_window') and layer_idx % 2 == 0) else 0
        
        # Q, K, V projections with GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, age=None, attn_mask=None):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q_shape = q.shape
        k_shape = k.shape
        q = _apply_rotary_emb(q.reshape(B * self.n_head, T, self.head_dim), cos, sin).reshape(q_shape)
        k = _apply_rotary_emb(k.reshape(B * self.n_kv_head, T, self.head_dim), cos, sin).reshape(k_shape)
        
        # Expand K, V to match number of query heads (GQA)
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k = k.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
        
        # Compute attention
        if self.flash and attn_mask is None:
            # Use Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            att = None
        else:
            # Manual attention with custom mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(causal_mask.view(1, 1, T, T), float('-inf'))
            
            # Sliding window mask
            if self.sliding_window > 0:
                window_mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=-self.sliding_window).bool()
                att = att.masked_fill(window_mask.view(1, 1, T, T), float('-inf'))
            
            # Custom medical attention mask
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        return y, att


class MixtureOfExperts(nn.Module):
    """Lightweight MoE for domain-specific medical knowledge"""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, 'num_experts') else 8
        self.experts_per_token = config.experts_per_token if hasattr(config, 'experts_per_token') else 2
        self.n_embd = config.n_embd
        self.intermediate_size = 4 * config.n_embd  # Standard FFN expansion
        
        # Router
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
        # Experts (smaller than in gpt-oss for efficiency)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias),
                nn.GELU(),
                nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout)
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        
        # Route tokens to experts
        router_logits = self.gate(x)  # (B, T, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.experts_per_token, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Compute expert outputs
        final_output = torch.zeros_like(x)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find all tokens that selected this expert at any position
            expert_mask = (selected_experts == i).any(dim=-1)  # (B, T)
            
            if not expert_mask.any():
                continue
            
            # Process all tokens that use this expert
            expert_input = x[expert_mask]  # (N, C) where N = expert_mask.sum()
            expert_output = self.experts[i](expert_input)  # (N, C)
            
            # Create a mapping to put expert_output back in the right places
            expert_output_full = torch.zeros_like(x)  # (B, T, C)
            expert_output_full[expert_mask] = expert_output
            
            # For each expert position k, add weighted contribution
            for k in range(self.experts_per_token):
                token_mask = (selected_experts[..., k] == i)  # (B, T)
                if token_mask.any():
                    weights = routing_weights[..., k:k+1]  # (B, T, 1)
                    final_output += weights * expert_output_full * token_mask.unsqueeze(-1)
        
        return final_output


class ModernFFN(nn.Module):
    """Modern FFN with SwiGLU activation"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        
        # SwiGLU requires 2 * intermediate_size for gating
        self.c_fc = nn.Linear(config.n_embd, 2 * self.intermediate_size, bias=config.bias)
        self.c_proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = swiglu(x, limit=7.0)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ModernBlock(nn.Module):
    """Transformer block with modern architecture"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        
        # Use MoE or standard FFN
        if hasattr(config, 'use_moe') and config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = ModernFFN(config)

    def forward(self, x, age=None, attn_mask=None):
        # Pre-norm architecture (more stable)
        y, att = self.attn(self.ln_1(x), age, attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att

@dataclass
class ModernDelphiConfig:
    """Configuration for Modern Delphi"""
    block_size: int = 1024
    vocab_size: int = 1290  # Death token: raw 1288 → shifted 1289
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4  # Grouped Query Attention (3x fewer KV heads)
    n_embd: int = 384
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False  # False is more modern
    
    # Medical specific
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    
    # Modern architecture features
    use_moe: bool = False  # Enable Mixture of Experts
    num_experts: int = 8
    experts_per_token: int = 2
    sliding_window: int = 256  # Sliding window attention
    rope_theta: float = 10000.0
    
    # Time-to-Event distribution: 'exponential' or 'weibull'
    time_distribution: str = 'exponential'

class ModernDelphi(nn.Module):
    """Modern Delphi with GPT-OSS inspired architecture"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wae = AgeEncoding(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ModernBlock(config, i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weibull shape head (for time-to-event)
        time_distribution = getattr(config, 'time_distribution', 'exponential')
        if time_distribution == 'weibull':
            self.time_shape_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

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
        
        # Token + Age embeddings
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age)
        
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)
        
        # Attention mask (medical specific)
        attn_mask = (idx > 0).view(b, 1, 1, t) * (idx > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1)))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        
        attn_mask = attn_mask + (idx == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        # Transformer blocks
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        
        x = self.transformer.ln_f(x)
        att = torch.stack(att) if att[0] is not None else None

        if targets is not None:
            # Medical loss computation (same as original Delphi)
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
                targets_flat[pass_tokens], 
                ignore_index=-1
            )
            
            # Time-to-event loss
            dt = torch.clamp(targets_age - age, min=1.0)
            
            if self.config.mask_ties:
                dt = torch.gather(
                    dt, -1, 
                    (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                    .max(-1).indices.squeeze((1, 2))
                )
            
            time_distribution = getattr(self.config, 'time_distribution', 'exponential')
            
            if time_distribution == 'weibull':
                # Weibull Distribution Loss (수치 안정성 강화)
                # Shape parameter (k > 0)
                time_shape = F.softplus(self.time_shape_head(x)) + 0.1  # (B, T, vocab_size)
                
                # λ = exp(logsumexp) ensures positivity
                log_scale = torch.logsumexp(logits, -1)  # (B, T)
                scale = torch.clamp(torch.exp(log_scale), min=0.1, max=1e4) + self.config.t_min
                
                # k = weighted mean shape (clamp for stability)
                event_probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size)
                shape = torch.clamp((event_probs * time_shape).sum(-1), min=0.1, max=10.0)  # (B, T)
                
                # Weibull negative log-likelihood with numerical stability
                dt_flat = torch.clamp(dt.view(-1), min=0.1) + self.config.t_min
                scale_flat = scale.view(-1)
                shape_flat = shape.view(-1)
                
                # Clamp ratio to prevent overflow
                ratio = torch.clamp(dt_flat / scale_flat, min=1e-6, max=1e3)
                
                log_likelihood = (
                    torch.log(shape_flat + 1e-8) - 
                    shape_flat * torch.log(scale_flat + 1e-8) + 
                    (shape_flat - 1) * torch.log(dt_flat + 1e-8) - 
                    ratio.pow(shape_flat)
                )
                # Clamp final loss to prevent NaN
                loss_dt = -torch.mean(torch.clamp(log_likelihood[pass_tokens], min=-100, max=100))
            else:
                # Exponential Distribution Loss (기존)
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
        """Configure optimizer (same as original Delphi)"""
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
                    # RMSNorm scale parameter
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # Any other weight parameter defaults to decay
                    decay.add(fpn)

        # Handle weight tying: lm_head.weight is tied to wte.weight (Embedding)
        # So it should be in no_decay, not decay
        if 'lm_head.weight' in decay:
            decay.discard('lm_head.weight')
        if 'lm_head.weight' not in no_decay:
            no_decay.add('lm_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Remove lm_head.weight from no_decay if it doesn't exist in param_dict (due to weight tying)
        # The actual parameter is wte.weight, which is already handled as an Embedding
        if 'lm_head.weight' in no_decay and 'lm_head.weight' not in param_dict:
            no_decay.discard('lm_head.weight')
        
        # Check for overlapping parameters and resolve
        inter_params = decay & no_decay
        if inter_params:
            if _is_master():
                print(f"Warning: Found {len(inter_params)} parameters in both decay and no_decay, moving to no_decay:")
                for pn in sorted(inter_params):
                    print(f"  {pn}")
            for pn in list(inter_params):
                decay.discard(pn)
        
        union_params = decay | no_decay
        
        # Find missing parameters
        missing_params = param_dict.keys() - union_params
        if missing_params:
            if _is_master():
                print(f"Warning: Found {len(missing_params)} unclassified parameters, adding to no_decay:")
                for pn in sorted(missing_params):
                    print(f"  {pn}")
            for pn in missing_params:
                no_decay.add(pn)
        
        # Final check
        union_params = decay | no_decay
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"Still have overlapping params: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"Missing params: {param_dict.keys() - union_params}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        if _is_master():
            print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, 
                 termination_tokens=None, top_k=None):
        """Generate medical trajectories (same as original Delphi)"""
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
            
            # Exponential sampling for time-to-event
            t_next = torch.clamp(
                -torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), 
                min=0, max=365*80
            ).min(1)
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
# Composite Embedding + Multi-Head Output Architecture
# =============================================================================

class CompositeEmbedding(nn.Module):
    """
    Composite Embedding Layer: 여러 입력 필드를 각각 임베딩하고 합산
    - DATA (약품/질병 코드) -> ID Embedding
    - SHIFT (시프트 값) -> Shift Embedding
    - TOTAL (기간) -> Duration Embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        # 각 필드별 Embedding
        self.data_emb = nn.Embedding(config.data_vocab_size, config.n_embd)
        self.shift_emb = nn.Embedding(config.shift_vocab_size, config.n_embd)
        self.total_emb = nn.Embedding(config.total_vocab_size, config.n_embd)
        
    def forward(self, data, shift, total):
        """
        Args:
            data: (B, T) DATA tokens
            shift: (B, T) SHIFT values (정수값)
            total: (B, T) TOTAL tokens
        Returns:
            combined embedding (B, T, n_embd)
        """
        # DATA embedding (clamp to valid range)
        data_idx = torch.clamp(data, min=0, max=self.data_emb.num_embeddings - 1)
        data_emb = self.data_emb(data_idx)
        
        # SHIFT embedding (clamp to valid range)
        shift_idx = torch.clamp(shift, min=0, max=self.shift_emb.num_embeddings - 1)
        shift_emb = self.shift_emb(shift_idx)
        
        # TOTAL embedding (clamp to valid range)
        total_idx = torch.clamp(total, min=0, max=self.total_emb.num_embeddings - 1)
        total_emb = self.total_emb(total_idx)
        
        # 모든 임베딩 합산
        combined = data_emb + shift_emb + total_emb
        
        return combined


class MultiHeadOutput(nn.Module):
    """
    Multi-Head Output Layer: 각 필드별 예측 헤드
    - DATA Head: 다음 DATA 토큰 예측 (Classification)
    - SHIFT Head: 다음 SHIFT 값 예측 (Classification)
    - TOTAL Head: 다음 TOTAL 값 예측 (Regression, 연속값)
    - Time Head: 다음 이벤트까지의 시간 예측
      - Exponential: scale (λ) parameter만 예측
      - Weibull: scale (λ) + shape (k) parameter 예측
    
    Drug-Conditioned Heads (optional):
    - 약물(drug) 정보를 조건으로 SHIFT/TOTAL 예측 성능 향상
    - FiLM (Feature-wise Linear Modulation) 방식 사용
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.time_distribution = getattr(config, 'time_distribution', 'exponential')
        
        # Drug-conditioning option
        self.use_drug_conditioning = getattr(config, 'use_drug_conditioning', False)
        
        # Classification Heads (DATA, SHIFT)
        self.data_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
        # Strengthened SHIFT Head: 2-layer MLP + LayerNorm for better classification
        # This gives the model more capacity to learn SHIFT patterns
        self.shift_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.shift_vocab_size)
        )
        
        # Regression Head (TOTAL) - 출력 dim = 1
        self.total_head = nn.Linear(config.n_embd, 1, bias=True)
        
        # ============================================================
        # Drug-Conditioned Heads (FiLM style)
        # 약물 정보로 hidden state를 변조하여 SHIFT/TOTAL 예측
        # ============================================================
        if self.use_drug_conditioning:
            # FiLM generator: drug_emb → (gamma, beta) for modulation
            # SHIFT: drug 조건
            self.shift_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)  # gamma, beta
            )
            self.shift_drug_cond_head = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 2),
                nn.GELU(),
                nn.Linear(config.n_embd // 2, config.shift_vocab_size)
            )
            
            # TOTAL: drug 조건만
            self.total_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)  # gamma, beta
            )
            self.total_drug_cond_head = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 2),
                nn.GELU(),
                nn.Linear(config.n_embd // 2, 1)
            )
            
            # Initialize FiLM generators for identity transformation (gamma=1, beta=0)
            # This ensures stable training start
            for film_gen in [self.shift_film_generator, self.total_film_generator]:
                last_layer = film_gen[-1]  # Last Linear layer
                nn.init.zeros_(last_layer.weight)
                # Initialize bias: first half (gamma) = 1, second half (beta) = 0
                with torch.no_grad():
                    last_layer.bias[:config.n_embd].fill_(1.0)  # gamma = 1
                    last_layer.bias[config.n_embd:].zero_()      # beta = 0
        
        # Time Head: scale parameter (λ) for all event types
        self.time_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
        # Weibull shape parameter (k) - 전역 또는 per-event
        if self.time_distribution == 'weibull':
            # Shape parameter per event type (more expressive)
            self.time_shape_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
    def forward(self, x, drug_emb=None, drug_token_mask=None):
        """
        Args:
            x: (B, T, n_embd) transformer output
            drug_emb: (B, T, n_embd) drug token embedding for conditioning (optional)
                      - 학습 시: INPUT data의 embedding (과거 정보)
                      - 추론 시: 현재까지의 data의 embedding
            drug_token_mask: (B, T) bool tensor, True where TARGET is a drug token
                             FiLM only applies at these positions
        Returns:
            dict of logits/values for each head
        """
        output = {
            'data': self.data_head(x),               # (B, T, data_vocab_size) - classification logits
            'shift': self.shift_head(x),             # (B, T, shift_vocab_size) - classification logits
            'total': self.total_head(x).squeeze(-1), # (B, T) - regression value
            'time_scale': self.time_head(x),         # (B, T, data_vocab_size) - λ parameter
        }
        
        # ============================================================
        # Drug-Conditioned Predictions (FiLM modulation)
        # Only applies when target is a drug token (drug_token_mask=True)
        # ============================================================
        if self.use_drug_conditioning and drug_emb is not None:
            # SHIFT: FiLM modulation with drug embedding
            shift_film = self.shift_film_generator(drug_emb)  # (B, T, n_embd*2)
            shift_gamma, shift_beta = shift_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            shift_modulated = shift_gamma * x + shift_beta  # FiLM: γ * x + β
            shift_drug_cond = self.shift_drug_cond_head(shift_modulated)  # (B, T, shift_vocab_size)
            
            # TOTAL: FiLM modulation with drug embedding
            total_film = self.total_film_generator(drug_emb)  # (B, T, n_embd*2)
            total_gamma, total_beta = total_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            total_modulated = total_gamma * x + total_beta  # FiLM: γ * x + β
            total_drug_cond = self.total_drug_cond_head(total_modulated).squeeze(-1)  # (B, T)
            
            # Apply drug token masking: only use FiLM output where target is a drug
            if drug_token_mask is not None:
                # Blend: use FiLM output for drug tokens, standard output otherwise
                # SHIFT: (B, T, shift_vocab_size)
                shift_drug_cond_masked = torch.where(
                    drug_token_mask.unsqueeze(-1),  # (B, T, 1)
                    shift_drug_cond,
                    output['shift']
                )
                output['shift_drug_cond'] = shift_drug_cond_masked
                
                # TOTAL: (B, T)
                total_drug_cond_masked = torch.where(
                    drug_token_mask,  # (B, T)
                    total_drug_cond,
                    output['total']
                )
                output['total_drug_cond'] = total_drug_cond_masked
            else:
                # No mask: apply FiLM to all positions (backward compatibility)
                output['shift_drug_cond'] = shift_drug_cond
                output['total_drug_cond'] = total_drug_cond
        
        # For backward compatibility
        output['time'] = output['time_scale']
        
        if self.time_distribution == 'weibull':
            # Weibull shape parameter (k > 0, use softplus to ensure positivity)
            output['time_shape'] = F.softplus(self.time_shape_head(x)) + 0.1  # (B, T, data_vocab_size)
        
        return output


@dataclass
class CompositeDelphiConfig:
    """Configuration for Composite Delphi with Multi-Head Output"""
    block_size: int = 1024
    
    # Vocabulary sizes for each field
    # 
    # Embedding vocab sizes (모든 필드의 embedding에 사용):
    # - DATA: includes drugs (Metformin~Death, raw 1277-1288) → after +1 shift: 1278-1289 → vocab_size = 1290
    # - SHIFT: range depends on dataset (need to check actual range)
    # - TOTAL: range 0-550 → vocab_size = 551
    #
    # Head 구조:
    # - DATA Head: Linear(n_embd, 1290) → Softmax + Cross-Entropy (Classification)
    # - SHIFT Head: Linear(n_embd, shift_vocab_size) → Softmax + Cross-Entropy (Classification)
    # - TOTAL Head: Linear(n_embd, 1) → Weighted Huber Loss (Regression)
    # Note: vocab sizes include +1 for the shift in get_batch_composite (0 reserved for padding)
    # Drug tokens: raw 1277~1288 → after +1 shift: 1278~1289 → max token 1289
    data_vocab_size: int = 1290   # DATA embedding & head (Classification) - includes Death token
    shift_vocab_size: int = 5     # SHIFT embedding & head (Classification) - values 0-3, after +1 shift: 1-4 → vocab_size=5
    total_vocab_size: int = 552   # TOTAL embedding only (Regression head dim=1) - max 551 after +1 shift
    
    # Model architecture
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    
    # Medical specific
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    
    # Drug-Conditioning: 약물 정보를 조건으로 SHIFT/TOTAL 예측 성능 향상
    # FiLM (Feature-wise Linear Modulation) 방식 사용
    use_drug_conditioning: bool = False
    
    # Drug token range (after +1 shift in get_batch_composite):
    # Raw tokens: Metformin(1278), Sulfonylurea(1279), DPP-4(1280), Insulin(1281), 
    #             Meglitinide(1282), Thiazolidinedione(1283), Alpha-glucosidase(1284),
    #             GLP-1(1285), SGLT-2(1286), Other(1287), Death(1288)
    # Shifted tokens: 1279-1289 (inclusive)
    drug_token_min: int = 1279  # First drug token after +1 shift (Metformin)
    drug_token_max: int = 1289  # Last drug token after +1 shift (Death)

    # Modern architecture features
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

    # SHIFT loss options (handles class imbalance if needed)
    # - shift_loss_type: 'ce' (weighted cross-entropy) or 'focal'
    # - shift_ignore_index: typically 0 (padding/unknown)
    # - shift_class_weights: per-class weights (length == shift_vocab_size). If empty, unweighted.
    shift_loss_type: str = 'focal'
    shift_ignore_index: int = 0
    shift_focal_gamma: float = 2.0
    shift_class_weights: list = field(default_factory=list)
    
    # Time-to-Event distribution: 'exponential' or 'weibull'
    # - exponential: 상수 hazard rate (memoryless)
    # - weibull: 시간에 따라 변하는 hazard rate (shape parameter k로 조절)
    time_distribution: str = 'exponential'


class CompositeDelphi(nn.Module):
    """
    Composite Delphi: Composite Embedding + Multi-Head Output
    
    입력: (DATA, SHIFT, TOTAL, AGE)
    출력: 각 필드별 예측 + 시간 예측
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Composite Embedding
        self.composite_emb = CompositeEmbedding(config)
        
        # Age Encoding (기존과 동일)
        self.age_encoding = AgeEncoding(config)
        
        # Dropout layers
        self.token_drop = nn.Dropout(config.token_dropout)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks (기존과 동일)
        self.h = nn.ModuleList([ModernBlock(config, i) for i in range(config.n_layer)])
        
        # Final normalization
        self.ln_f = RMSNorm(config.n_embd)
        
        # Multi-Head Output
        self.multi_head = MultiHeadOutput(config)
        
        # Weight tying: data_head와 data_emb
        self.multi_head.data_head.weight = self.composite_emb.data_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
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
        """
        Args:
            data: (B, T) DATA tokens
            shift: (B, T) SHIFT values
            total: (B, T) TOTAL tokens
            age: (B, T) AGE values
            targets_*: 각 필드의 타겟 (optional)
        """
        device = data.device
        b, t = data.size()
        
        # 1. Composite Embedding
        composite_emb = self.composite_emb(data, shift, total)
        
        # 2. Age Encoding
        age_emb = self.age_encoding(age)
        
        # 3. Combine embeddings
        x = self.token_drop(composite_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.drop(x)
        
        # 4. Attention mask
        attn_mask = (data > 0).view(b, 1, 1, t) * (data > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        if targets_data is not None and self.config.mask_ties:
            attn_mask *= (age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        
        attn_mask = attn_mask + (data == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        # 5. Transformer blocks
        att_list = []
        for block in self.h:
            x, att = block(x, age, attn_mask)
            att_list.append(att)
        
        x = self.ln_f(x)
        att = torch.stack(att_list) if att_list[0] is not None else None
        
        # 6. Multi-Head Output
        # Drug-Conditioning: FiLM modulation for SHIFT/TOTAL prediction
        # CRITICAL FIX: 
        #   - Use input `data` (current tokens), NOT `targets_data` (future tokens)
        #   - This prevents information leakage during training
        #   - FiLM learns to modulate predictions based on past drug history
        drug_emb = None
        drug_token_mask = None
        if self.config.use_drug_conditioning:
            # Get drug embedding from INPUT data (not targets!)
            drug_source = data  # Current input tokens
            if drug_source is not None:
                # Clamp to valid range
                drug_source_clamped = torch.clamp(
                    drug_source,
                    min=0,
                    max=self.composite_emb.data_emb.num_embeddings - 1,
                )
                drug_emb = self.composite_emb.data_emb(drug_source_clamped)
            
            # Create drug token mask: FiLM only applies when TARGET is a drug
            # Drug tokens range (after +1 shift): 1279-1289
            drug_token_min = getattr(self.config, 'drug_token_min', 1279)
            drug_token_max = getattr(self.config, 'drug_token_max', 1289)
            if targets_data is not None:
                drug_token_mask = (targets_data >= drug_token_min) & (targets_data <= drug_token_max)
        
        logits = self.multi_head(x, drug_emb=drug_emb, drug_token_mask=drug_token_mask)
        
        # 7. Compute losses if targets provided
        if targets_data is not None:
            loss = self._compute_loss(
                logits, data, age,
                targets_data, targets_shift, targets_total, targets_age,
                attn_mask, validation_loss_mode
            )
        else:
            loss = None
        
        return logits, loss, att
    
    def _compute_loss(self, logits, data, age,
                      targets_data, targets_shift, targets_total, targets_age,
                      attn_mask, validation_loss_mode):
        """Compute multi-head losses"""
        device = data.device
        b, t = data.size()
        
        ignored_tokens = self.config.ignore_tokens.copy()
        if validation_loss_mode:
            ignored_tokens += [1]
        
        # Valid token mask
        targets_flat = targets_data.reshape(-1)
        pass_tokens = targets_flat != -1
        for k in ignored_tokens:
            pass_tokens = pass_tokens * (targets_flat != k)
        
        # Clamp targets to valid vocab range (defensive measure)
        data_vocab_size = self.config.data_vocab_size
        targets_flat_clamped = torch.clamp(targets_flat, min=0, max=data_vocab_size - 1)
        
        # 1. DATA Cross-Entropy Loss
        data_logits = logits['data']
        if validation_loss_mode:
            data_logits[..., ignored_tokens] = -torch.inf
        
        loss_data = F.cross_entropy(
            data_logits.reshape(-1, data_logits.size(-1))[pass_tokens],
            targets_flat_clamped[pass_tokens],  # ← clamp된 값 사용
            ignore_index=-1
        )
        
        # 2. SHIFT Classification Loss (Weighted CE or Focal) + ignore padding(0)
        shift_logits_source = logits['shift']
        if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
            shift_logits_source = logits['shift_drug_cond']

        shift_targets_all = targets_shift.reshape(-1)
        shift_pass_tokens = shift_targets_all != -1

        shift_logits_flat = shift_logits_source.reshape(-1, shift_logits_source.size(-1))[shift_pass_tokens]
        shift_targets_flat = shift_targets_all[shift_pass_tokens]
        
        # Clamp targets to valid vocab range (defensive measure)
        shift_vocab_size = self.config.shift_vocab_size
        shift_targets_flat = torch.clamp(shift_targets_flat, min=-1, max=shift_vocab_size - 1)
        
        shift_ignore = int(getattr(self.config, 'shift_ignore_index', 0))
        # also ignore explicit -1 if present
        shift_valid = (shift_targets_flat != -1) & (shift_targets_flat != shift_ignore)
        shift_logits_flat = shift_logits_flat[shift_valid]
        shift_targets_flat = shift_targets_flat[shift_valid]

        if shift_logits_flat.numel() == 0:
            loss_shift = torch.tensor(0.0, device=device)
        else:
            shift_loss_type = str(getattr(self.config, 'shift_loss_type', 'ce')).lower()
            weights_list = getattr(self.config, 'shift_class_weights', None)
            weight_t = None
            if isinstance(weights_list, (list, tuple)) and len(weights_list) == self.config.shift_vocab_size:
                weight_t = torch.tensor(weights_list, device=device, dtype=torch.float32)

            if shift_loss_type == 'focal':
                gamma = float(getattr(self.config, 'shift_focal_gamma', 2.0))
                loss_shift = focal_loss_multiclass(
                    shift_logits_flat,
                    shift_targets_flat.long(),
                    gamma=gamma,
                    alpha=weight_t,
                    ignore_index=None,  # already filtered
                    reduction='mean',
                )
            else:
                # weighted cross-entropy (if weight_t is provided)
                loss_shift = F.cross_entropy(
                    shift_logits_flat,
                    shift_targets_flat.long(),
                    weight=weight_t,
                    ignore_index=-1,
                )
        
        # 3. TOTAL Regression Loss (redesigned for better gradient flow)
        # Problem: normalized 0-1 scale produced tiny gradients (~0.0002)
        # Solution: Use raw scale (0-550) with MSE loss and asymmetric weighting
        targets_total_float = targets_total.float()
        total_target = targets_total_float.reshape(-1)[pass_tokens]  # (N,) target, raw scale
        
        # ============================================================
        # Drug-Conditioned TOTAL Loss (if available)
        # ============================================================
        if 'total_drug_cond' in logits and self.config.use_drug_conditioning:
            # Use drug-conditioned prediction
            total_pred = logits['total_drug_cond'].reshape(-1)[pass_tokens]
        else:
            # Fallback: standard TOTAL head
            total_pred = logits['total'].reshape(-1)[pass_tokens]
        
        # Clamp predictions to valid range (raw scale 0-550)
        total_scale = 550.0
        total_pred = torch.clamp(total_pred, min=0.0, max=total_scale)
        
        # Asymmetric weighting: non-zero targets get 10x weight
        # This prevents model from just predicting 0 for everything
        nonzero_mask = total_target > 0
        zero_mask = ~nonzero_mask
        
        # MSE Loss (larger gradients than Huber on small values)
        mse_all = (total_pred - total_target) ** 2
        
        # Compute weighted loss
        if nonzero_mask.sum() > 0 and zero_mask.sum() > 0:
            loss_nonzero = mse_all[nonzero_mask].mean()
            loss_zero = mse_all[zero_mask].mean()
            # Non-zero targets are much more important (10x weight)
            loss_total = 0.9 * loss_nonzero + 0.1 * loss_zero
        elif nonzero_mask.sum() > 0:
            loss_total = mse_all[nonzero_mask].mean()
        else:
            loss_total = mse_all.mean()
        
        # Normalize by scale^2 to keep loss in reasonable range
        # This makes loss ~1.0 instead of ~300000.0
        loss_total = loss_total / (total_scale ** 2)
        
        # 4. Time-to-Event Loss
        # dt = time difference (days until next event)
        dt = torch.clamp(targets_age - age, min=1.0)
        
        if self.config.mask_ties:
            dt = torch.gather(
                dt, -1,
                (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                .max(-1).indices.squeeze((1, 2))
            )
        
        time_distribution = getattr(self.config, 'time_distribution', 'exponential')
        
        if time_distribution == 'weibull':
            # ============================================================
            # Weibull Distribution Loss (수치 안정성 강화 v2)
            # ============================================================
            # PDF: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
            # log f(t) = log(k) - k*log(λ) + (k-1)*log(t) - (t/λ)^k
            
            # Use time head logits for scale parameter (consistent with exponential)
            time_logits = logits['time_scale']  # (B, T, data_vocab_size)
            time_shape = logits['time_shape']  # (B, T, vocab_size) - k (positive)
            
            # λ = exp(logsumexp) ensures positivity
            log_scale = torch.logsumexp(time_logits, -1)  # (B, T)
            scale = torch.clamp(torch.exp(log_scale), min=1.0, max=365.0)  # 1일~1년 범위로 제한
            
            # k = weighted mean shape (clamp for stability)
            event_probs = F.softmax(time_logits, dim=-1)  # (B, T, vocab_size)
            shape = torch.clamp((event_probs * time_shape).sum(-1), min=0.5, max=5.0)  # 더 좁은 범위
            
            # Weibull negative log-likelihood with numerical stability
            dt_flat = torch.clamp(dt.view(-1), min=1.0)  # 최소 1일
            scale_flat = scale.view(-1)
            shape_flat = shape.view(-1)
            
            # Compute log-likelihood components separately for numerical stability
            log_shape = torch.log(shape_flat)
            log_scale = torch.log(scale_flat)
            log_dt = torch.log(dt_flat)
            
            # (dt/scale)^k = exp(k * log(dt/scale))
            log_ratio = log_dt - log_scale  # log(dt/scale)
            power_term = torch.exp(shape_flat * log_ratio)  # (dt/scale)^k
            
            # Clamp power term to prevent explosion
            power_term = torch.clamp(power_term, max=50.0)
            
            log_likelihood = (
                log_shape - 
                shape_flat * log_scale + 
                (shape_flat - 1) * log_dt - 
                power_term
            )
            
            # Negative log-likelihood (loss)
            # Clamp to reasonable range and only use valid tokens
            nll = -log_likelihood[pass_tokens]
            loss_time = torch.clamp(nll, min=0.0, max=20.0).mean()
            
        else:
            # ============================================================
            # Exponential Distribution Loss (기존 Delphi와 동일)
            # ============================================================
            # PDF: f(t) = λ * exp(-λt)
            # log f(t) = log(λ) - λt
            
            # Use time head logits for time calculation (original Delphi approach)
            time_logits = logits['time_scale']  # (B, T, data_vocab_size)
            lse = torch.logsumexp(time_logits, -1)  # (B, T)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_time = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_time = torch.mean(loss_time[pass_tokens])
        
        # Weighted sum of losses
        total_loss = (
            self.config.loss_weight_data * loss_data +
            self.config.loss_weight_shift * loss_shift +
            self.config.loss_weight_total * loss_total +
            self.config.loss_weight_time * loss_time
        )
        
        return {
            'loss': total_loss,
            'loss_data': loss_data,
            'loss_shift': loss_shift,
            'loss_total': loss_total,
            'loss_time': loss_time
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer"""
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
                    # RMSNorm scale parameter
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # Any other weight parameter defaults to decay
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Handle weight tying: multi_head.data_head.weight is tied to composite_emb.data_emb.weight (Embedding)
        # The actual parameter is composite_emb.data_emb.weight, which is already in no_decay (Embedding)
        # So we just need to remove multi_head.data_head.weight from decay if it exists
        if 'multi_head.data_head.weight' in decay:
            decay.discard('multi_head.data_head.weight')
        # Don't add it to no_decay if it doesn't exist in param_dict (due to weight tying)
        
        # Check for overlapping parameters and resolve
        inter_params = decay & no_decay
        if inter_params:
            if _is_master():
                print(f"Warning: Found {len(inter_params)} parameters in both decay and no_decay, moving to no_decay:")
                for pn in sorted(inter_params):
                    print(f"  {pn}")
            for pn in list(inter_params):
                decay.discard(pn)
        
        union_params = decay | no_decay
        
        # Find missing parameters
        missing_params = param_dict.keys() - union_params
        if missing_params:
            if _is_master():
                print(f"Warning: Found {len(missing_params)} unclassified parameters, adding to no_decay:")
                for pn in sorted(missing_params):
                    print(f"  {pn}")
            for pn in missing_params:
                no_decay.add(pn)
        
        # Final check - only check params that actually exist
        union_params = decay | no_decay
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"Still have overlapping params: {inter_params}"
        
        # Only check params that exist in param_dict
        missing_in_union = param_dict.keys() - union_params
        assert len(missing_in_union) == 0, f"Missing params: {missing_in_union}"
        
        # Only include params that exist in param_dict
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, data, shift, total, age, 
                 max_new_tokens=100, max_age=85*365.25,
                 no_repeat=True, termination_tokens=None):
        """Generate composite sequences"""
        if termination_tokens is None:
            warnings.warn('Consider setting termination_tokens for your dataset.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=data.device)
        
        if max_new_tokens == -1:
            max_new_tokens = 128
        
        for _ in range(max_new_tokens):
            logits, _, _ = self(data, shift, total, age, drug_conditioning_data=data)
            
            # Get last position logits
            data_logits = logits['data'][:, -1, :]
            shift_logits_source = logits['shift']
            if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
                shift_logits_source = logits['shift_drug_cond']
            shift_logits = shift_logits_source[:, -1, :]

            total_pred_source = logits['total']
            if 'total_drug_cond' in logits and self.config.use_drug_conditioning:
                total_pred_source = logits['total_drug_cond']
            total_pred = total_pred_source[:, -1]
            time_logits = logits['time'][:, -1, :]
            
            # Mask ignored tokens
            data_logits[:, self.config.ignore_tokens] = -torch.inf
            
            if no_repeat:
                fill = data.clone()
                fill[fill == 1] = 0
                data_logits = data_logits.scatter_(1, fill, -torch.inf)
            
            # Sample next tokens
            # Data: exponential sampling for time-to-event
            t_next = torch.clamp(
                -torch.exp(-time_logits) * torch.rand(time_logits.shape, device=data.device).log(),
                min=0, max=365*80
            ).min(1)
            
            data_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]
            
            # Sample shift, total from their distributions
            shift_next = torch.argmax(shift_logits, dim=-1, keepdim=True)
            total_next = (
                torch.clamp(
                    total_pred.round(),
                    min=0,
                    max=self.config.total_vocab_size - 1,
                )
                .long()
                .unsqueeze(-1)
            )
            
            # Append to sequences
            data = torch.cat((data, data_next), dim=1)
            shift = torch.cat((shift, shift_next), dim=1)
            total = torch.cat((total, total_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # Check termination
            if torch.logical_or(
                torch.isin(data, termination_tokens).any(-1), 
                age_next > max_age
            ).all():
                break
        
        return data, shift, total, age, logits
