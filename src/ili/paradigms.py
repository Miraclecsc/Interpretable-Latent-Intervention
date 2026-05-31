"""Bridging cached latents into the model's input embeddings.

Both continuous-thought paradigms evaluated in the paper expose the same
decode-time interface: a fixed-length span of continuous thoughts is placed in
the prefix and the answer is then decoded autoregressively.

* **CoConut** reserves a ``<|latent|>`` span inside the prompt; cached thoughts
  are written onto those reserved slots.
* **CODI** distills a CoT into a compact sequence of continuous vectors that are
  prefixed to the input; the same reserved-slot mechanism is used here so that
  cached CODI thoughts occupy the latent span.

Because the cached vectors are paradigm-agnostic at write time, a single helper
covers both; the paradigm choice only documents how the cache was produced.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def build_prefix(loaded, input_ids: List[int]) -> Tuple[torch.Tensor, List[int]]:
    """Embed ``input_ids`` and return ``(inputs_embeds, latent_positions)``."""
    ids = torch.tensor([input_ids], device=loaded.device, dtype=torch.long)
    inputs_embeds = loaded.model.get_input_embeddings()(ids)
    positions = (ids[0] == loaded.latent_id).nonzero(as_tuple=False).view(-1).tolist()
    return inputs_embeds, positions


def write_latents(inputs_embeds: torch.Tensor, latent_positions: List[int],
                  chain: torch.Tensor) -> torch.Tensor:
    """Overwrite the reserved latent slots with cached continuous thoughts."""
    n = min(len(latent_positions), chain.shape[0])
    chain = chain.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
    for j in range(n):
        inputs_embeds[0, latent_positions[j], :] = chain[j]
    return inputs_embeds


@torch.no_grad()
def prompt_query_hidden(loaded, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Final-layer hidden state at the end of the prefix (retrieval query).

    Used by the Answer-Anchored Slerp to retrieve an exemplar (Appendix C.2).
    """
    out = loaded.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].detach().float()
