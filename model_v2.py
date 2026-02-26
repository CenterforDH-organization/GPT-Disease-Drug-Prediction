"""
Composite Delphi v2
- Adds FPG-aware conditioning for drug-conditioned SHIFT/TOTAL predictions.
- Uses existing model components from model.py without modifying them.
"""

from dataclasses import dataclass, field

import torch

from model import (
    CompositeDelphi,
    CompositeDelphiConfig,
)


@dataclass
class CompositeDelphiV2Config(CompositeDelphiConfig):
    """Composite Delphi v2 config with FPG conditioning options."""

    use_fpg_conditioning: bool = True
    fpg_token_ids: list = field(default_factory=lambda: [19, 20, 21])
    fpg_condition_scale: float = 1.0


class CompositeDelphiV2(CompositeDelphi):
    """
    Composite Delphi v2: Adds FPG token conditioning to drug FiLM pathway.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_fpg_conditioning = getattr(config, 'use_fpg_conditioning', True)
        self.fpg_condition_scale = float(getattr(config, 'fpg_condition_scale', 1.0))
        fpg_token_ids = getattr(config, 'fpg_token_ids', [19, 20, 21])
        self.register_buffer(
            'fpg_token_ids_tensor',
            torch.tensor(fpg_token_ids, dtype=torch.long),
            persistent=False,
        )

    def _compute_fpg_context(self, data):
        """
        Compute per-sequence FPG context embedding from DATA tokens.

        Returns:
            fpg_context: (B, n_embd)
        """
        if not self.use_fpg_conditioning or self.fpg_token_ids_tensor.numel() == 0:
            return torch.zeros(data.size(0), self.config.n_embd, device=data.device)

        # Create mask for FPG tokens in the input sequence
        fpg_mask = torch.isin(data, self.fpg_token_ids_tensor)

        if not fpg_mask.any():
            return torch.zeros(data.size(0), self.config.n_embd, device=data.device)

        data_idx = torch.clamp(
            data,
            min=0,
            max=self.composite_emb.data_emb.num_embeddings - 1,
        )
        data_emb = self.composite_emb.data_emb(data_idx)
        fpg_mask_f = fpg_mask.unsqueeze(-1).float()

        masked_sum = (data_emb * fpg_mask_f).sum(dim=1)
        counts = fpg_mask_f.sum(dim=1).clamp(min=1.0)
        fpg_context = masked_sum / counts
        return fpg_context * self.fpg_condition_scale

    def forward(
        self,
        data,
        shift,
        total,
        age,
        targets_data=None,
        targets_shift=None,
        targets_total=None,
        targets_age=None,
        drug_conditioning_data=None,
        validation_loss_mode=False,
    ):
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
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
                torch.ones(t, device=device)
            ) > 0

        attn_mask = attn_mask + (data == 0).view(b, 1, 1, t) * torch.diag(
            torch.ones(t, device=device)
        ) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0

        # 5. Transformer blocks
        att_list = []
        for block in self.h:
            x, att = block(x, age, attn_mask)
            att_list.append(att)

        x = self.ln_f(x)
        att = torch.stack(att_list) if att_list[0] is not None else None

        # 6. Multi-Head Output with drug-conditioning + FPG conditioning
        drug_emb = None
        drug_token_mask = None
        if self.config.use_drug_conditioning:
            drug_source = data  # Current input tokens
            if drug_source is not None:
                drug_source_clamped = torch.clamp(
                    drug_source,
                    min=0,
                    max=self.composite_emb.data_emb.num_embeddings - 1,
                )
                drug_emb = self.composite_emb.data_emb(drug_source_clamped)

            if self.use_fpg_conditioning and drug_emb is not None:
                fpg_context = self._compute_fpg_context(data)
                drug_emb = drug_emb + fpg_context.unsqueeze(1)

            drug_token_min = getattr(self.config, 'drug_token_min', 1279)
            drug_token_max = getattr(self.config, 'drug_token_max', 1289)
            if targets_data is not None:
                drug_token_mask = (targets_data >= drug_token_min) & (targets_data <= drug_token_max)

        logits = self.multi_head(x, drug_emb=drug_emb, drug_token_mask=drug_token_mask)

        # 7. Compute losses if targets provided
        if targets_data is not None:
            loss = self._compute_loss(
                logits,
                data,
                age,
                targets_data,
                targets_shift,
                targets_total,
                targets_age,
                attn_mask,
                validation_loss_mode,
            )
        else:
            loss = None

        return logits, loss, att


__all__ = [
    'CompositeDelphiV2',
    'CompositeDelphiV2Config',
]
