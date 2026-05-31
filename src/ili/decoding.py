"""Greedy decoding from a prefilled ``inputs_embeds`` prefix.

The latent span is supplied as continuous embeddings (not token ids), so we feed
``inputs_embeds`` directly and autoregressively append the embedding of each
greedily-selected token. This mirrors the decoding contract of the trained
latent-reasoning models and keeps the answer-parsing convention (split on '#').
"""

from __future__ import annotations

from typing import List, Tuple

import torch


@torch.no_grad()
def greedy_decode(model, inputs_embeds: torch.Tensor, eos_token_id: int,
                  max_new_tokens: int = 128) -> List[int]:
    """Greedy-decode ``max_new_tokens`` ids starting from ``inputs_embeds``."""
    emb = model.get_input_embeddings()
    out = model(inputs_embeds=inputs_embeds)
    next_id = int(torch.argmax(out.logits[0, -1]))
    tokens = [next_id]
    cur = torch.cat(
        [inputs_embeds, emb(torch.tensor([[next_id]], device=inputs_embeds.device))], dim=1)
    for _ in range(max_new_tokens - 1):
        out = model(inputs_embeds=cur)
        next_id = int(torch.argmax(out.logits[0, -1]))
        tokens.append(next_id)
        if next_id == eos_token_id:
            break
        cur = torch.cat(
            [cur, emb(torch.tensor([[next_id]], device=cur.device))], dim=1)
    return tokens


def extract_prediction(text: str) -> str:
    """Parse the final answer from generated text (convention: after '#')."""
    return text.split("#")[-1].replace(",", "").strip()


def decode_and_predict(loaded, input_ids: List[int], inputs_embeds: torch.Tensor,
                       max_new_tokens: int = 128) -> Tuple[str, str]:
    """Decode from ``inputs_embeds`` and return (full_text, parsed_prediction)."""
    gen = greedy_decode(loaded.model, inputs_embeds,
                        loaded.tokenizer.eos_token_id, max_new_tokens)
    full = torch.cat([torch.tensor([input_ids], device=loaded.device),
                      torch.tensor([gen], device=loaded.device)], dim=1)
    text = loaded.tokenizer.decode(full[0], skip_special_tokens=True)
    return text, extract_prediction(text)
