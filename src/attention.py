import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Scaled Dot-Product Attention with optional causal masking.
    
    attention_mask is expected as (B, Lk) with 1=valid, 0=padding

    Args:
        mask_future (bool): If True, applies a causal mask that blocks 
                           attention to later positions.
    """

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask=None):
        # Detect head dimension presence
        added_head_dim = False
        if query.dim() == 3:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            added_head_dim = True
        elif query.dim() != 4:
            raise ValueError(f"Attention.forward expects query of dim 3 or 4; "
                             f"got {query.dim()}")

        B, H, Lq, D = query.shape
        _, _, Lk, _ = key.shape

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)

        # Causal mask
        if self.mask_future:
            causal_mask = torch.triu(torch.ones(1,
                                                1,
                                                Lq,
                                                Lk,
                                                dtype=torch.bool,
                                                device=scores.device),
                                     diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Padding mask
        if attention_mask is not None:
            key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value)

        if added_head_dim:
            context = context.squeeze(1)

        return context


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer using the Attention class.
    
    Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        mask_future (bool): If True, applies causal masking.
    """

    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(mask_future=mask_future)

    def forward(self, query, key, value, attention_mask=None):
        B, Lq, D = query.shape
        Lk = key.shape[1]

        # Linear projections
        q_proj = self.query_transform(query)
        k_proj = self.key_transform(key)
        v_proj = self.value_transform(value)

        # Split heads: (B, H, L, D_head)
        q_proj = q_proj.view(B, Lq, self.num_heads,
                             self.d_head).transpose(1, 2)
        k_proj = k_proj.view(B, Lk, self.num_heads,
                             self.d_head).transpose(1, 2)
        v_proj = v_proj.view(B, Lk, self.num_heads,
                             self.d_head).transpose(1, 2)

        # Apply attention
        context = self.attention(q_proj, k_proj, v_proj, attention_mask)

        # Merge heads: (B, Lq, D)
        context = context.transpose(1, 2).contiguous().view(B, Lq, D)

        # Output linear
        output = self.output_transform(context)
        return output
