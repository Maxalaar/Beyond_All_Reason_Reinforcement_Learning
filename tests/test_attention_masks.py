"""Masked attention of the architecture: padding rows, and no branch on the data (needed by the ONNX export)."""

import torch

from beyond_all_reason_reinforcement_learning.architecture.category_aggregation import CategoryAggregation
from beyond_all_reason_reinforcement_learning.architecture.entity_encoder import EntityEncoder

torch.manual_seed(0)


def entity_encoder() -> EntityEncoder:
    return EntityEncoder(
        n_chassis=3,
        n_state=4,
        n_weapon=5,
        d_level1=8,
        entity_projector_hidden_sizes=[8],
        weapon_projector_hidden_sizes=[8],
        number_entity_attention_heads=2,
    ).eval()


def test_an_entity_without_weapon_is_encoded_as_with_no_weapon_slot():
    encoder = entity_encoder()
    chassis, state, weapons = torch.randn(3, 3), torch.randn(3, 4), torch.randn(3, 2, 5)
    weapon_mask = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    with torch.no_grad():
        tokens = encoder(chassis, state, weapons, weapon_mask)
        without_slots = encoder(chassis, state, torch.zeros(3, 0, 5), torch.zeros(3, 0))

    assert torch.isfinite(tokens).all()
    assert torch.allclose(tokens[1], without_slots[1])
    assert not torch.allclose(tokens[0], without_slots[0])


def test_a_category_without_entity_gives_a_zero_token_and_leaves_the_other_rows_unchanged():
    aggregation = CategoryAggregation(d_level1=8, d_level2=8, number_category_attention_heads=2).eval()
    tokens = torch.randn(2, 3, 8)
    with torch.no_grad():
        together = aggregation(tokens, torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        alone = aggregation(tokens[:1], torch.tensor([[1.0, 0.0, 1.0]]))

    assert torch.equal(together[1], torch.zeros(8))
    assert torch.allclose(together[0], alone[0])


def test_the_masked_attentions_export_with_a_free_batch_size():
    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.export.export(
        entity_encoder(),
        (torch.randn(4, 3), torch.randn(4, 4), torch.randn(4, 2, 5), torch.ones(4, 2)),
        dynamic_shapes=({0: batch}, {0: batch}, {0: batch}, {0: batch}),
    )
    torch.export.export(
        CategoryAggregation(d_level1=8, d_level2=8, number_category_attention_heads=2).eval(),
        (torch.randn(4, 3, 8), torch.ones(4, 3)),
        dynamic_shapes=({0: batch}, {0: batch}),
    )


def test_each_query_given_asks_the_category_its_own_question():
    aggregation = CategoryAggregation(d_level1=8, d_level2=8, number_category_attention_heads=2, learned_query=False)
    assert not hasattr(aggregation, "learned_query")
    tokens, mask = torch.randn(2, 3, 8), torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    queries = torch.randn(2, 2, 8)
    with torch.no_grad():
        answers = aggregation(tokens, mask, queries)
        swapped = aggregation(tokens, mask, queries.flip(1))

    assert answers.shape == (2, 2, 8)
    assert not torch.allclose(answers[0, 0], answers[0, 1])
    assert torch.allclose(answers[0], swapped[0].flip(0))
    assert torch.equal(answers[1], torch.zeros(2, 8))
