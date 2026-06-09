"""Compact Transformer encoder for binary entailment."""

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderSpec:
    vocab_size: int
    max_length: int
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    num_classes: int = 2
    logic_feature_dim: int = 8

    def to_dict(self) -> Dict[str, int | float]:
        return asdict(self)


class EntailmentTransformer(nn.Module):
    """Token, position, and segment embeddings with a Transformer encoder."""

    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(
            spec.vocab_size,
            spec.embedding_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            spec.max_length,
            spec.embedding_dim,
        )
        self.segment_embedding = nn.Embedding(2, spec.embedding_dim)
        self.embedding_norm = nn.LayerNorm(spec.embedding_dim)
        self.embedding_dropout = nn.Dropout(spec.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.embedding_dim,
            nhead=spec.num_heads,
            dim_feedforward=spec.hidden_dim,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spec.num_layers,
            norm=nn.LayerNorm(spec.embedding_dim),
            enable_nested_tensor=False,
        )
        self.logic_projection = nn.Sequential(
            nn.Linear(spec.logic_feature_dim, spec.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(spec.embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(spec.embedding_dim * 2, spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, spec.num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        logic_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, sequence_length)

        embeddings = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(segment_ids)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))
        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=~attention_mask.bool(),
        )
        structured = self.logic_projection(logic_features)
        combined = torch.cat((encoded[:, 0], structured), dim=1)
        return self.classifier(combined)
