import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings with Multimodal Fusion:
        - Token Embeddings: item ID
        - Position Embeddings: 위치 정보
        - Multimodal Fusion: Text + Image + Category features
    """
    def __init__(self, vocab_size: int, embed_size: int, max_len: int, 
                 dropout_rate: float = 0.3,
                 text_dim: int = 0, image_dim: int = 0, category_dim: int = 0,
                 use_multimodal: bool = False):
        super(BERTEmbeddings, self).__init__()

        self.use_multimodal = use_multimodal
        self.embed_size = embed_size

        # ── 1. 기본 embedding ────────────────────────────────────────────
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_len, embed_size)

        # ── 2. 멀티모달 projection ────────────────────────────────────────
        if use_multimodal:
            self.use_text = text_dim > 0
            self.use_image = image_dim > 0
            self.use_category = category_dim > 0

            if self.use_text:
                self.text_proj = nn.Linear(text_dim, embed_size)
            if self.use_image:
                self.image_proj = nn.Linear(image_dim, embed_size)
            if self.use_category:
                self.category_proj = nn.Linear(category_dim, embed_size)

            # Fusion layer (concat all modalities → embed_size)
            num_modalities = (1 
                            + int(self.use_text) 
                            + int(self.use_image) 
                            + int(self.use_category))
            self.fusion_linear = nn.Linear(embed_size * num_modalities, embed_size)
            self.fusion_norm = nn.LayerNorm(embed_size, eps=1e-6)

        # ── 3. Normalization + Dropout ────────────────────────────────────
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def _multimodal_fusion(self, seq, text_feat=None, image_feat=None, category_feat=None):
        """
        멀티모달 피처 fusion (SASRec 방식과 동일)
        
        Args:
            seq: (B, T) - item indices
            text_feat: (B, T, text_dim)
            image_feat: (B, T, image_dim)
            category_feat: (B, T, category_dim)
        
        Returns:
            fused: (B, T, embed_size)
        """
        parts = []

        # 1) Item ID embedding
        token_emb = self.token_embeddings(seq)
        token_emb = token_emb * math.sqrt(self.embed_size)  # scaling
        parts.append(token_emb)

        # 2) Text
        if self.use_text and text_feat is not None:
            parts.append(self.text_proj(text_feat))

        # 3) Image
        if self.use_image and image_feat is not None:
            parts.append(self.image_proj(image_feat))

        # 4) Category
        if self.use_category and category_feat is not None:
            parts.append(self.category_proj(category_feat))

        # Concat → Fusion
        fused = torch.cat(parts, dim=-1)  # (B, T, embed_size * num_modalities)
        fused = self.fusion_linear(fused)  # (B, T, embed_size)
        fused = self.fusion_norm(fused)

        return fused
    
    def forward(self, seq: torch.Tensor, segment_label: Optional[torch.Tensor] = None,
                text_feat=None, image_feat=None, category_feat=None) -> torch.Tensor:
        """
        Args:
            seq: (B, T)
            text_feat: (B, T, text_dim) or None
            image_feat: (B, T, image_dim) or None
            category_feat: (B, T, category_dim) or None
        
        Returns:
            embeddings: (B, T, embed_size)
        """
        batch_size, seq_length = seq.size()

        # Position indices
        position_ids = torch.arange(seq_length, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Token + Position embeddings
        if self.use_multimodal:
            # Multimodal fusion
            embeddings = self._multimodal_fusion(seq, text_feat, image_feat, category_feat)
        else:
            # 기본 방식 (ID만)
            embeddings = self.token_embeddings(seq)

        # Add positional embeddings
        embeddings = embeddings + self.positional_embeddings(position_ids)

        # LayerNorm + Dropout
        return self.dropout(self.layer_norm(embeddings))


class MultiHeadedAttention(nn.Module):
    """Multi-Head Self Attention"""

    def __init__(self, head_num, hidden_dim, dropout_rate_attn=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert hidden_dim % head_num == 0

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        # Q, K, V projection
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(p=dropout_rate_attn)

        # output projection
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        # Q, K, V 생성
        query = self.query_linear(q)
        key   = self.key_linear(k)
        value = self.value_linear(v)

        # (B, T, D) → (B, H, T, D/H)
        query = query.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        key   = key.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        # Attention score
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale

        # Padding mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention 확률
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Weighted sum
        out = torch.matmul(attention, value)

        # (B, T, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.hidden_dim)

        return self.output_linear(out), attention


class SublayerConnection(nn.Module):
    """Residual + LayerNorm + Dropout"""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, sublayer_out):
        return x + self.dropout(self.layer_norm(sublayer_out))


class PositionwiseFeedForward(nn.Module):
    """FFN: (D → 4D → D)"""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerEncoder(nn.Module):
    """Transformer Block = Attention + FFN"""

    def __init__(self, hidden_dim, head_num, ff_dim, dropout_rate=0.1, dropout_rate_attn=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadedAttention(head_num, hidden_dim, dropout_rate_attn)

        self.input_sublayer = SublayerConnection(hidden_dim, dropout_rate)
        self.feed_forward   = PositionwiseFeedForward(hidden_dim, ff_dim, dropout_rate)
        self.output_sublayer = SublayerConnection(hidden_dim, dropout_rate)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, mask)

        # Residual 1
        x = self.input_sublayer(x, attn_out)

        # FFN + residual
        x = self.output_sublayer(x, self.feed_forward(x))

        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT4Rec with Multimodal Fusion (BERT4RecF+)
    """

    def __init__(
        self,
        vocab_size=30522,
        max_len=512,
        hidden_dim=768,
        encoder_num=12,
        head_num=12,
        dropout_rate=0.1,
        dropout_rate_attn=0.1,
        initializer_range=0.02,
        # Multimodal
        text_dim=0,
        image_dim=0,
        category_dim=0,
        use_multimodal=False
    ):
        super(BERT, self).__init__()

        self.ff_dim = hidden_dim * 4
        self.use_multimodal = use_multimodal

        # Embedding layer (with multimodal fusion)
        self.embedding = BERTEmbeddings(
            vocab_size, hidden_dim, max_len, dropout_rate,
            text_dim=text_dim,
            image_dim=image_dim,
            category_dim=category_dim,
            use_multimodal=use_multimodal
        )

        # Transformer stack
        self.transformers = nn.ModuleList([
            TransformerEncoder(hidden_dim, head_num, self.ff_dim,
                               dropout_rate, dropout_rate_attn)
            for _ in range(encoder_num)
        ])

        # Weight initialization
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq, segment_info=None, 
                text_feat=None, image_feat=None, category_feat=None):
        """
        Args:
            seq: (B, T)
            text_feat: (B, T, text_dim) or None
            image_feat: (B, T, image_dim) or None
            category_feat: (B, T, category_dim) or None

        Returns:
            (B, T, hidden_dim)
        """
        # Padding mask
        mask = (seq > 0).unsqueeze(1).unsqueeze(1)

        # Embedding (with multimodal fusion)
        x = self.embedding(seq, segment_info, text_feat, image_feat, category_feat)

        # Transformer encoder stack
        for transformer in self.transformers:
            x = transformer(x, mask)

        return x
