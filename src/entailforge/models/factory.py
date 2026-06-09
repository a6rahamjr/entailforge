"""Model construction helpers."""

from entailforge.models.encoder import EncoderSpec, EntailmentTransformer
from entailforge.utils.config import AppConfig


def build_model(config: AppConfig, vocab_size: int) -> EntailmentTransformer:
    spec = EncoderSpec(
        vocab_size=vocab_size,
        max_length=config.data.max_length,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes,
    )
    return EntailmentTransformer(spec)
