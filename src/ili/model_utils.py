"""Loading the base causal LM and the trained latent-reasoning checkpoint.

The continuous-thought paradigms (CoConut / CODI) add three special tokens to
mark the latent span and fine-tune the backbone. At inference we reload the
backbone, register those tokens, and merge the trained weights. The latent
vectors themselves are read from cache, so no latent-generation forward pass is
required here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

START_LATENT = "<|start-latent|>"
END_LATENT = "<|end-latent|>"
LATENT = "<|latent|>"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: torch.dtype
    start_id: int
    end_id: int
    latent_id: int
    special_ids: List[int] = field(default_factory=list)


def _merge_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Merge a trained latent-reasoning checkpoint into the bare backbone.

    Handles the ``base_causallm.`` key prefix used by CoConut-style training and
    resizes the embedding table when the checkpoint vocabulary differs.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state

    tgt_vocab = None
    for key in ("base_causallm.model.embed_tokens.weight",
                "model.embed_tokens.weight", "transformer.wte.weight",
                "embed_tokens.weight"):
        if key in state:
            tgt_vocab = state[key].shape[0]
            break
    if tgt_vocab is None:
        for k, v in state.items():
            if k.endswith("embed_tokens.weight"):
                tgt_vocab = v.shape[0]
                break

    cur_vocab = model.get_input_embeddings().weight.shape[0]
    if tgt_vocab and tgt_vocab != cur_vocab:
        model.resize_token_embeddings(tgt_vocab)

    if any(k.startswith("base_causallm.") for k in state.keys()):
        state = {k.replace("base_causallm.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def load_model(model_base: str, ckpt_path: str = None, use_bf16: bool = True,
               device: str = None) -> LoadedModel:
    """Load the backbone, register latent tokens, and optionally merge a ckpt."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    for t in (START_LATENT, END_LATENT, LATENT):
        tok.add_tokens(t)
    start_id = tok.convert_tokens_to_ids(START_LATENT)
    end_id = tok.convert_tokens_to_ids(END_LATENT)
    latent_id = tok.convert_tokens_to_ids(LATENT)

    model = AutoModelForCausalLM.from_pretrained(
        model_base, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    model.resize_token_embeddings(len(tok))

    # Initialize the new special-token rows from an existing token so the
    # untrained slots are well-conditioned before the checkpoint is merged.
    with torch.no_grad():
        emb = model.get_input_embeddings()
        fallback = tok.convert_tokens_to_ids("<<")
        if fallback is None or fallback == tok.unk_token_id or fallback < 0:
            fallback = tok.eos_token_id
        for tid in (latent_id, start_id, end_id):
            emb.weight[tid] = emb.weight[fallback]
            if hasattr(model, "lm_head"):
                model.lm_head.weight[tid] = model.lm_head.weight[fallback]

    if ckpt_path:
        _merge_checkpoint(model, ckpt_path)

    return LoadedModel(model=model, tokenizer=tok, device=device, dtype=dtype,
                       start_id=start_id, end_id=end_id, latent_id=latent_id,
                       special_ids=[start_id, end_id, latent_id])
