import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Union, List

from attention import Attention, MultiHeadAttention
from util import get_device

@dataclass
class TransformerConfig:
    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6

    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000

    max_src_len: int = 512
    max_tgt_len: int = 512
    learn_pos_embed: bool = False

    
class PositionalEncoding(nn.Module):
    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
        requires_grad: bool
    """
    
    def __init__(self, max_len: int, embed_dim: int, requires_grad: bool = False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

    def _build_positional_encodings(self):
        encoding = torch.zeros(
            self.max_len, self.embed_dim, dtype=torch.float
        )
        position_idx = torch.arange(
            0, self.max_len, dtype=torch.float
        ).reshape(-1, 1)
        embed_dim_skip_idx = torch.arange(
            0, self.embed_dim, step=2, dtype=torch.float
        )
        
        encoding[:, 0::2] = torch.sin(
            position_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim))
        )
        encoding[:, 1::2] = torch.cos(
            position_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim))
        )

        encoding = nn.Parameter(encoding, requires_grad=self.requires_grad)
        return encoding

    def forward(self, x: torch.Tensor):
        # Grab shape of tensor
        seq_len = x.shape[1]

        # Clip encodings to length needed
        encodings = self.encodings[:seq_len]

        # Add positional embeddings to data and return
        return x + encodings


class Embeddings(nn.Module):
    """
    All the embeddings we need for the source and target language. 
    Both source and target need:

    - Token Embeddings
    - Positional Embeddings
    """
    
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.src_embeddings = nn.Embedding(
            config.src_vocab_size, config.embedding_dimension
        )
        self.tgt_embeddings = nn.Embedding(
            config.tgt_vocab_size, config.embedding_dimension
        )
        
        self.src_positional_encodings = PositionalEncoding(
            config.max_src_len, 
            config.embedding_dimension, 
            config.learn_pos_embed
        )
        self.tgt_positional_encodings = PositionalEncoding(
            config.max_tgt_len, 
            config.embedding_dimension, 
            config.learn_pos_embed
        )
        
    def forward_src(self, input_ids):
        x = self.src_embeddings(input_ids)
        return self.src_positional_encodings(x)
    
    def forward_tgt(self, input_ids):
        x = self.tgt_embeddings(input_ids)
        return self.tgt_positional_encodings(x)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.embedding_dimension * config.mlp_ratio

        self.fc1 = nn.Linear(config.embedding_dimension, hidden)
        self.fc2 = nn.Linear(hidden, config.embedding_dimension)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, config):
        super().__init__()

        dim = config.embedding_dimension
        dropout = config.hidden_dropout_p

        self.self_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            d_model=dim,
            num_heads=config.num_attention_heads,
            mask_future=False
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(config)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # ---- Multi-Head Self-Attention block ----
        y = self.self_attn_norm(x)
        y = self.self_attn_dropout(
            self.self_attn(y, y, y, attention_mask))
        x = x + y  # residual

        # ---- Feed-Forward block ----
        y = self.ffn_dropout(
            self.ffn(
            self.ffn_norm(x)))
        x = x + y  # residual

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Causal attention + cross-attention + MLP layer
    """
    
    def __init__(self, config):
        super().__init__()
    
        self.dec_attention = MultiHeadAttention(
            d_model=config.embedding_dimension,
            num_heads=config.num_attention_heads,
            mask_future=True
        )
        self.dec_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.dec_attention_layernorm = nn.LayerNorm(
            config.embedding_dimension
        )
    
        self.cross_attention = MultiHeadAttention(
            d_model=config.embedding_dimension,
            num_heads=config.num_attention_heads,
            mask_future=False
        )
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(
            config.embedding_dimension
        )
    
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        tgt = tgt + self.dec_attention_dropout(
            self.dec_attention(tgt, tgt, tgt, attention_mask=tgt_mask)
        )
        tgt = self.dec_attention_layernorm(tgt)
    
        tgt = tgt + self.cross_attention_dropout(
            self.cross_attention(tgt, src, src, attention_mask=src_mask)
        )
        tgt = self.cross_attention_layernorm(tgt)
    
        tgt = tgt + self.feed_forward(tgt)
        return self.final_layer_norm(tgt)


class Transformer(nn.Module):
    """
    Final Transformer architecture proposed in Attention is All You Need
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.encodings = Embeddings(config)
    
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.encoder_depth)
        ])
    
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(config.decoder_depth)
        ])
    
        self.head = nn.Linear(
            config.embedding_dimension, config.tgt_vocab_size
        )
    
        # Initialize weights
        self.apply(self._init_weights_)
    
    def forward(
        self, src_ids, tgt_ids, 
        src_attention_mask=None, tgt_attention_mask=None
    ):
        src_embeddings = self.encodings.forward_src(src_ids)
        tgt_embeddings = self.encodings.forward_tgt(tgt_ids)
        
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_attention_mask)
        
        for layer in self.decoder:
            tgt_embeddings = layer(
                src_embeddings, tgt_embeddings,
                src_mask=src_attention_mask, 
                tgt_mask=tgt_attention_mask
            )
        
        return self.head(tgt_embeddings)
    
    @torch.no_grad()
    def inference(
        self, src_ids: torch.Tensor, tgt_start_id: int = 2, tgt_end_id: int = 3, max_len: int = 512
        ) -> List[int]:
        tgt_ids = torch.tensor([[tgt_start_id]], device=src_ids.device)
        src_embeddings = self.encodings.forward_src(src_ids)
        
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
        
        for _ in range(max_len):
            tgt_embeddings = self.encodings.forward_tgt(tgt_ids)
            for layer in self.decoder:
                tgt_embeddings = layer(src_embeddings, tgt_embeddings)
            
            last_step = tgt_embeddings[:, -1, :]
            pred = self.head(last_step).argmax(dim=-1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids, pred], dim=1)
            
            if torch.all(pred == tgt_end_id):
                break
        
        return tgt_ids.squeeze().cpu().tolist()

    def _init_weights_(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


if __name__ == "__main__":
    device = get_device()
    src = torch.randint(0, 1000, (1, 128)).to(device)
    config = TransformerConfig()
    model = Transformer(config)
    model = model.to(device)
    model.inference(src)
