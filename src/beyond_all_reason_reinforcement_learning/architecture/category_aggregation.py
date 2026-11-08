"""Level 2 of the hierarchical attention architecture: aggregate N entity tokens into fixed-size tokens.

Queries attend over the entity tokens produced by Level 1, compressing a variable-length (and masked) set of entities
into one token per query. The queries are given by the caller (the actor: made from the observing unit's token, so
each unit asks the category what matters to it), or one learned query vector (the centralized critic, which observes
from no unit).

When N = 0 (the maximum for that category is 0 in the configuration) or all entities in a
batch element are padding, the output is a zero tensor so that the Level 3 attention is not
influenced by absent categories.
"""

import torch
import torch.nn as nn

from beyond_all_reason_reinforcement_learning.architecture.entity_encoder import attention_safe_mask


class CategoryAggregation(nn.Module):
    def __init__(
        self,
        d_level1: int,
        d_level2: int,
        number_category_attention_heads: int,
        learned_query: bool = True,
    ):
        super().__init__()
        assert d_level2 % number_category_attention_heads == 0, (
            f"d_level2 ({d_level2}) must be divisible by "
            f"number_category_attention_heads ({number_category_attention_heads})"
        )
        # project entity tokens (d_level1) → key / value space (d_level2)
        self.key_value_projector = nn.Linear(d_level1, d_level2)
        # One learned query vector, when the caller gives none.
        if learned_query:
            self.learned_query = nn.Parameter(torch.randn(d_level2))
        self.attention = nn.MultiheadAttention(
            embed_dim=d_level2,
            num_heads=number_category_attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.d_level2 = d_level2

    def forward(
        self,
        entity_tokens: torch.Tensor,  # (B, N, d_level1)
        entity_mask: torch.Tensor,  # (B, N) — 1.0 = real entity, 0.0 = padding
        queries: torch.Tensor | None = None,  # (B, K, d_level2), or None: the learned query
    ) -> torch.Tensor:  # (B, K, d_level2) for given queries, (B, d_level2) for the learned one
        B, N, _ = entity_tokens.shape
        learned = queries is None
        if learned:
            queries = self.learned_query.expand(B, 1, -1)  # (B, 1, d_level2)
        K = queries.shape[1]

        if N == 0:
            tokens = torch.zeros(B, K, self.d_level2, device=entity_tokens.device, dtype=entity_tokens.dtype)
        else:
            kv = self.key_value_projector(entity_tokens)  # (B, N, d_level2)
            # Rows with no real entity: zero (see attention_safe_mask).
            safe_mask, all_padding = attention_safe_mask(entity_mask == 0.0)
            tokens, _ = self.attention(queries, kv, kv, key_padding_mask=safe_mask)
            tokens = tokens.masked_fill(all_padding[:, None, None], 0.0)  # (B, K, d_level2)
        return tokens.squeeze(1) if learned else tokens
