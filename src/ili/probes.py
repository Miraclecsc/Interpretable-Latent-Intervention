"""Interpretability probes that supply the priors for the interventions.

These correspond to the diagnostic tools of Section 3 and Appendix B:

* :class:`Mapper`  -- the linear/MLP map f_phi that reconstructs an explicit
  CoT step representation from a latent surrogate (Section 3.2). It provides the
  *semantic* prior used by Intervention A.
* :class:`EnergyMLP` -- the scalar energy function H trained with a margin
  ranking loss (Section 3.4, Eq. 11). It provides the *geometric* prior used by
  Intervention C.2.
* :func:`build_embedding_subspace_projector` -- the projector P_r onto the top
  principal components of the (tied) embedding matrix, used as the manifold
  constraint Omega for Interventions A and C.
* :func:`weight_tying_expected_embedding` -- the expected token embedding
  E^T softmax(o / tau) used by Intervention C.1 (Eq. 15).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mapper(nn.Module):
    """Latent-to-CoT mapper f_phi (Section 3.2, Appendix B.1).

    ``num_layers == 1`` degenerates to a single linear map (the "linear
    recoverability" probe). ``num_layers == 2`` adds a ReLU + LayerNorm hidden
    block, matching the trained checkpoints used in the experiments.
    """

    def __init__(self, dim: int, hidden_dim: int = 1024, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers <= 1 or hidden_dim <= 0:
            self.net = nn.Sequential(nn.Linear(dim, dim))
        else:
            layers = [nn.Linear(dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_checkpoint(cls, path: str, dim: int, hidden_dim: int = 1024,
                        num_layers: int = 2, device: str = "cpu") -> "Mapper":
        """Load a mapper saved by ``scripts/train_mapper.py``.

        The checkpoint stores ``{"model": state_dict, "meta": {...}}``; the meta
        block overrides the architecture hyper-parameters when present.
        """
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        dim = int(meta.get("d", dim))
        hidden_dim = int(meta.get("hidden_dim", hidden_dim))
        num_layers = int(meta.get("num_layers", num_layers))
        model = cls(dim, hidden_dim, num_layers, float(meta.get("dropout", 0.0)))
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.to(device=device, dtype=torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


class EnergyMLP(nn.Module):
    """Scalar energy function H over hidden states (Section 3.4, Eq. 11).

    Trained so that energy strictly decreases along the latent chain from the
    last question token to the first answer token. Stores feature whitening
    statistics so the same normalization is applied at decode time.
    """

    def __init__(self, dim: int, hidden_dim: int = 1024, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        mods, d = [], dim
        for _ in range(max(1, num_layers)):
            mods += [nn.Linear(d, hidden_dim), nn.ReLU()]
            if dropout > 0:
                mods += [nn.Dropout(dropout)]
            d = hidden_dim
        mods += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*mods)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / (self.sigma + 1e-6)
        return self.net(x).squeeze(-1)

    @classmethod
    def from_checkpoint(cls, path: str, dim: int, hidden_dim: int = 1024,
                        num_layers: int = 2, device: str = "cpu") -> "EnergyMLP":
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = cls(dim, hidden_dim, num_layers)
        model.load_state_dict(state, strict=False)
        model.to(device=device, dtype=torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


@torch.no_grad()
def build_embedding_subspace_projector(embedding_weight: torch.Tensor, rank: int) -> Optional[torch.Tensor]:
    """Build P_r = V_r V_r^T onto the top-``rank`` PCs of the embedding matrix.

    ``embedding_weight`` is the [V, d] (tied) input embedding. Returns a [d, d]
    projection matrix on CPU in float32, or ``None`` when ``rank <= 0``.
    This realizes the Embedding-Subspace Alignment constraint Omega (Section 4.2).
    """
    if rank <= 0:
        return None
    E = embedding_weight.detach().float().cpu()
    E = E - E.mean(dim=0, keepdim=True)
    d = E.shape[1]
    r = min(rank, d)
    # Low-rank PCA is sufficient and far cheaper than a full SVD for large V.
    q = min(max(r + 8, r), min(E.shape[0] - 1, d))
    _, _, Vt = torch.pca_lowrank(E, q=q)
    Vr = Vt[:, :r]                      # [d, r]
    return (Vr @ Vr.T).contiguous()     # [d, d]


def project_to_subspace(z: torch.Tensor, projector: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply the embedding-subspace projector P_r to ``z`` (Omega for A/C)."""
    if projector is None:
        return z
    return projector.to(z.dtype).to(z.device) @ z


@torch.no_grad()
def weight_tying_expected_embedding(logits: torch.Tensor, embedding_weight: torch.Tensor,
                                    tau: float = 1.0) -> torch.Tensor:
    """Expected token embedding under the tied output head (Eq. 15).

        e_tilde = E^T softmax(o / tau)

    ``logits`` is the [V] vector of logits produced at the position *before* the
    latent slot; ``embedding_weight`` is the [V, d] tied embedding matrix.
    """
    p = F.softmax(logits.float() / max(tau, 1e-6), dim=-1)
    return (p @ embedding_weight.float())
