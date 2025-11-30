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

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings


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
        
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i)  # (B, T, experts_per_token)
            expert_mask = expert_mask.any(dim=-1)  # (B, T)
            
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = self.experts[i](expert_input)
                
                # Weight by routing probability
                for k in range(self.experts_per_token):
                    token_mask = (selected_experts[..., k] == i)
                    weights = routing_weights[..., k:k+1][token_mask]
                    final_output[token_mask] += weights * expert_output
        
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
    vocab_size: int = 1270 ############ NEED TO MODIFY
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
            lse = torch.logsumexp(logits, -1)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            
            if self.config.mask_ties:
                dt = torch.gather(
                    dt, -1, 
                    (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                    .max(-1).indices.squeeze((1, 2))
                )
            
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
        
        # Check for overlapping parameters and resolve
        inter_params = decay & no_decay
        if inter_params:
            print(f"Warning: Found {len(inter_params)} parameters in both decay and no_decay, moving to no_decay:")
            for pn in sorted(inter_params):
                print(f"  {pn}")
                decay.discard(pn)
        
        union_params = decay | no_decay
        
        # Find missing parameters
        missing_params = param_dict.keys() - union_params
        if missing_params:
            print(f"Warning: Found {len(missing_params)} unclassified parameters, adding to no_decay:")
            for pn in sorted(missing_params):
                print(f"  {pn}")
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
    - DOSE (용량) -> Dose Embedding (이산화된 값)
    - TOTAL (기간) -> Duration Embedding
    - UNIT (단위) -> Unit Embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        # 각 필드별 Embedding
        self.data_emb = nn.Embedding(config.data_vocab_size, config.n_embd)
        self.dose_emb = nn.Embedding(config.dose_vocab_size, config.n_embd)
        self.total_emb = nn.Embedding(config.total_vocab_size, config.n_embd)
        self.unit_emb = nn.Embedding(config.unit_vocab_size, config.n_embd)
        
        # Dose는 연속값일 수 있으므로 이산화용 버킷 경계 정의
        # 예: [0, 0.5, 1, 2, 5, 10, 20, 50, 100, ...]
        self.register_buffer('dose_buckets', torch.tensor(
            [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        ))
        
    def discretize_dose(self, dose):
        """연속 dose 값을 이산 버킷 인덱스로 변환"""
        # dose가 각 버킷보다 큰지 확인하여 버킷 인덱스 계산
        bucket_idx = (dose.unsqueeze(-1) > self.dose_buckets).sum(-1)
        return bucket_idx.long()
        
    def forward(self, data, dose, total, unit):
        """
        Args:
            data: (B, T) DATA tokens
            dose: (B, T) DOSE values (연속 또는 이산)
            total: (B, T) TOTAL tokens
            unit: (B, T) UNIT tokens
        Returns:
            combined embedding (B, T, n_embd)
        """
        # DATA embedding
        data_emb = self.data_emb(data)
        
        # DOSE embedding (이산화)
        if dose.dtype == torch.float32 or dose.dtype == torch.float64:
            dose_idx = self.discretize_dose(dose)
        else:
            dose_idx = dose
        dose_emb = self.dose_emb(dose_idx)
        
        # TOTAL embedding
        total_emb = self.total_emb(total)
        
        # UNIT embedding
        unit_emb = self.unit_emb(unit)
        
        # 모든 임베딩 합산
        combined = data_emb + dose_emb + total_emb + unit_emb
        
        return combined


class MultiHeadOutput(nn.Module):
    """
    Multi-Head Output Layer: 각 필드별 예측 헤드
    - ID Head: 다음 DATA 토큰 예측
    - Dose Head: 다음 DOSE 값 예측
    - Duration Head: 다음 TOTAL 값 예측
    - Unit Head: 다음 UNIT 값 예측
    - Time Head: 다음 이벤트까지의 시간 예측
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        # 각 필드별 Linear Head
        self.data_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        self.dose_head = nn.Linear(config.n_embd, config.dose_vocab_size, bias=False)
        self.total_head = nn.Linear(config.n_embd, config.total_vocab_size, bias=False)
        self.unit_head = nn.Linear(config.n_embd, config.unit_vocab_size, bias=False)
        
        # Time Head는 모든 vocab에 대한 rate parameter 출력
        self.time_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) transformer output
        Returns:
            dict of logits for each head
        """
        return {
            'data': self.data_head(x),      # (B, T, data_vocab_size)
            'dose': self.dose_head(x),      # (B, T, dose_vocab_size)
            'total': self.total_head(x),    # (B, T, total_vocab_size)
            'unit': self.unit_head(x),      # (B, T, unit_vocab_size)
            'time': self.time_head(x),      # (B, T, data_vocab_size)
        }


@dataclass
class CompositeDelphiConfig:
    """Configuration for Composite Delphi with Multi-Head Output"""
    block_size: int = 1024
    
    # Vocabulary sizes for each field
    data_vocab_size: int = 1500  # 약품/질병 코드 수
    dose_vocab_size: int = 16    # 이산화된 dose 버킷 수
    total_vocab_size: int = 128  # 기간 (일수) vocab
    unit_vocab_size: int = 8     # 단위 vocab
    
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
    
    # Modern architecture features
    use_moe: bool = True
    num_experts: int = 8
    experts_per_token: int = 2
    sliding_window: int = 256
    rope_theta: float = 10000.0
    
    # Loss weights
    loss_weight_data: float = 1.0
    loss_weight_dose: float = 0.5
    loss_weight_total: float = 0.5
    loss_weight_unit: float = 0.5
    loss_weight_time: float = 1.0


class CompositeDelphi(nn.Module):
    """
    Composite Delphi: Composite Embedding + Multi-Head Output
    
    입력: (DATA, DOSE, TOTAL, UNIT, AGE)
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
        
        print(f"CompositeDelphi parameters: {self.get_num_params()/1e6:.2f}M")
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, data, dose, total, unit, age,
                targets_data=None, targets_dose=None, targets_total=None, 
                targets_unit=None, targets_age=None,
                validation_loss_mode=False):
        """
        Args:
            data: (B, T) DATA tokens
            dose: (B, T) DOSE values
            total: (B, T) TOTAL tokens
            unit: (B, T) UNIT tokens
            age: (B, T) AGE values
            targets_*: 각 필드의 타겟 (optional)
        """
        device = data.device
        b, t = data.size()
        
        # 1. Composite Embedding
        composite_emb = self.composite_emb(data, dose, total, unit)
        
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
        logits = self.multi_head(x)
        
        # 7. Compute losses if targets provided
        if targets_data is not None:
            loss = self._compute_loss(
                logits, data, age,
                targets_data, targets_dose, targets_total, targets_unit, targets_age,
                attn_mask, validation_loss_mode
            )
        else:
            loss = None
        
        return logits, loss, att
    
    def _compute_loss(self, logits, data, age,
                      targets_data, targets_dose, targets_total, targets_unit, targets_age,
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
        
        # 1. DATA Cross-Entropy Loss
        data_logits = logits['data']
        if validation_loss_mode:
            data_logits[..., ignored_tokens] = -torch.inf
        
        loss_data = F.cross_entropy(
            data_logits.reshape(-1, data_logits.size(-1))[pass_tokens],
            targets_flat[pass_tokens],
            ignore_index=-1
        )
        
        # 2. DOSE Cross-Entropy Loss
        if targets_dose.dtype == torch.float32 or targets_dose.dtype == torch.float64:
            targets_dose_idx = self.composite_emb.discretize_dose(targets_dose)
        else:
            targets_dose_idx = targets_dose
        
        loss_dose = F.cross_entropy(
            logits['dose'].reshape(-1, logits['dose'].size(-1))[pass_tokens],
            targets_dose_idx.reshape(-1)[pass_tokens],
            ignore_index=-1
        )
        
        # 3. TOTAL Cross-Entropy Loss
        loss_total = F.cross_entropy(
            logits['total'].reshape(-1, logits['total'].size(-1))[pass_tokens],
            targets_total.reshape(-1)[pass_tokens],
            ignore_index=-1
        )
        
        # 4. UNIT Cross-Entropy Loss
        loss_unit = F.cross_entropy(
            logits['unit'].reshape(-1, logits['unit'].size(-1))[pass_tokens],
            targets_unit.reshape(-1)[pass_tokens],
            ignore_index=-1
        )
        
        # 5. Time-to-Event Loss (기존 Delphi와 동일)
        time_logits = logits['time']
        lse = torch.logsumexp(time_logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        dt = torch.clamp(targets_age - age, min=1.0)
        
        if self.config.mask_ties:
            dt = torch.gather(
                dt, -1,
                (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                .max(-1).indices.squeeze((1, 2))
            )
        
        ldt = -torch.log(dt + self.config.t_min).view(-1)
        loss_time = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
        loss_time = torch.mean(loss_time[pass_tokens])
        
        # Weighted sum of losses
        total_loss = (
            self.config.loss_weight_data * loss_data +
            self.config.loss_weight_dose * loss_dose +
            self.config.loss_weight_total * loss_total +
            self.config.loss_weight_unit * loss_unit +
            self.config.loss_weight_time * loss_time
        )
        
        return {
            'loss': total_loss,
            'loss_data': loss_data,
            'loss_dose': loss_dose,
            'loss_total': loss_total,
            'loss_unit': loss_unit,
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
        
        # Handle weight tying: multi_head.data_head.weight is tied to composite_emb.data_emb.weight (Embedding)
        # So it should be in no_decay, not decay
        if 'multi_head.data_head.weight' in decay:
            decay.discard('multi_head.data_head.weight')
        if 'multi_head.data_head.weight' not in no_decay:
            no_decay.add('multi_head.data_head.weight')
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Check for overlapping parameters and resolve
        inter_params = decay & no_decay
        if inter_params:
            print(f"Warning: Found {len(inter_params)} parameters in both decay and no_decay, moving to no_decay:")
            for pn in sorted(inter_params):
                print(f"  {pn}")
                decay.discard(pn)
        
        union_params = decay | no_decay
        
        # Find missing parameters
        missing_params = param_dict.keys() - union_params
        if missing_params:
            print(f"Warning: Found {len(missing_params)} unclassified parameters, adding to no_decay:")
            for pn in sorted(missing_params):
                print(f"  {pn}")
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
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, data, dose, total, unit, age, 
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
            logits, _, _ = self(data, dose, total, unit, age)
            
            # Get last position logits
            data_logits = logits['data'][:, -1, :]
            dose_logits = logits['dose'][:, -1, :]
            total_logits = logits['total'][:, -1, :]
            unit_logits = logits['unit'][:, -1, :]
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
            
            # Sample dose, total, unit from their distributions
            dose_next = torch.argmax(dose_logits, dim=-1, keepdim=True)
            total_next = torch.argmax(total_logits, dim=-1, keepdim=True)
            unit_next = torch.argmax(unit_logits, dim=-1, keepdim=True)
            
            # Append to sequences
            data = torch.cat((data, data_next), dim=1)
            dose = torch.cat((dose, dose_next.float()), dim=1)
            total = torch.cat((total, total_next), dim=1)
            unit = torch.cat((unit, unit_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # Check termination
            if torch.logical_or(
                torch.isin(data, termination_tokens).any(-1), 
                age_next > max_age
            ).all():
                break
        
        return data, dose, total, unit, age, logits

