"""Access to pre-computed (cached) latent artifacts.

The latent generation pass that produces the continuous thoughts is run *once*
and stored on disk; this repository's interventions consume those caches so that
inference is fully training-free. Expected on-disk layout::

    latent_root/
        case_00000/ latent_01.pt ... latent_0K.pt        # cached thoughts z_1..z_K
        case_00001/ ...
    cot_root/                                             # only for probe training
        case_00000/ step_end_01.pt step_end_02.pt answer_start.pt

Each ``*.pt`` file is a 1-D hidden-state tensor of size ``d``. Missing cases
return ``None`` so callers can fall back to plain (un-intervened) decoding.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch


def _load_vec(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.detach().to(torch.float32).cpu()
    return t[0] if t.ndim == 2 else t  # accept [1, d] or [d]


def load_latent_chain(latent_root: str, case_idx: int, num_latent: int = 6) -> Optional[torch.Tensor]:
    """Load the cached chain ``z_1..z_K`` for a case as a [K, d] tensor."""
    vecs = []
    case_dir = os.path.join(latent_root, f"case_{case_idx:05d}")
    for j in range(1, num_latent + 1):
        p = os.path.join(case_dir, f"latent_{j:02d}.pt")
        if not os.path.exists(p):
            return None
        vecs.append(_load_vec(p))
    return torch.stack(vecs, dim=0)


def load_cot_targets(cot_root: str, case_idx: int) -> dict:
    """Load CoT step-boundary hidden states for a case (probe supervision)."""
    case_dir = os.path.join(cot_root, f"case_{case_idx:05d}")
    out = {}
    for name in ("step_end_01", "step_end_02", "answer_start"):
        p = os.path.join(case_dir, f"{name}.pt")
        if os.path.exists(p):
            out[name] = _load_vec(p)
    return out


class AnchorRetriever:
    """Top-1 exemplar retrieval for the Answer-Anchored Slerp (Appendix C.2).

    Builds a bank of the *final* latent thoughts ``z_K`` of cached training
    exemplars. At query time the prompt's final hidden state is matched against
    the bank by cosine similarity; the top-1 exemplar's final latent thought is
    returned as the anchor ``t`` used to steer the causal hub.
    """

    def __init__(self, anchors: torch.Tensor):
        # anchors: [N, d]; stored L2-normalized for fast cosine lookup.
        self.raw = anchors
        self.normed = anchors / (anchors.norm(dim=-1, keepdim=True) + 1e-12)

    @classmethod
    def from_cache(cls, latent_root: str, case_indices: List[int],
                   num_latent: int = 6) -> "AnchorRetriever":
        bank = []
        for i in case_indices:
            chain = load_latent_chain(latent_root, i, num_latent)
            if chain is not None:
                bank.append(chain[-1])  # final latent thought z_K
        if not bank:
            raise RuntimeError(f"No exemplar latents found under {latent_root}")
        return cls(torch.stack(bank, dim=0))

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Return the anchor ``t`` (raw vector) most similar to ``query``."""
        q = (query.float() / (query.float().norm() + 1e-12)).cpu()
        sims = self.normed @ q
        return self.raw[int(torch.argmax(sims))]
