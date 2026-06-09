import torch

from entailforge.models.encoder import EncoderSpec, EntailmentTransformer


def test_model_forward_and_backward():
    model = EntailmentTransformer(
        EncoderSpec(
            vocab_size=32,
            max_length=12,
            embedding_dim=16,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(1, 32, (4, 12))
    attention_mask = torch.ones((4, 12), dtype=torch.bool)
    segment_ids = torch.zeros((4, 12), dtype=torch.long)
    logic_features = torch.zeros((4, 8), dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1])

    logits = model(input_ids, attention_mask, segment_ids, logic_features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (4, 2)
    assert any(parameter.grad is not None for parameter in model.parameters())
