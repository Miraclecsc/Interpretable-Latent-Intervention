"""Training-free, decode-time interventions on latent reasoning states.

Every intervention operates on the *cached* latent chain that has already been
written into the prefix ``inputs_embeds`` at the positions ``latent_positions``.
None of them update model parameters; they only edit the continuous thought
vectors before greedy decoding of the final answer.

All variants are instances of the unified update rule of Algorithm 1:

    v*      = Phi(z_k)                          # extract guidance signal
    d       = v* - z_k
    z_steer = z_k + alpha * (d / ||d||) * ||z_k||
    z'_k    = (1 - lam) * z_steer + lam * Omega(z_steer)

with the guidance prior ``Phi`` and the manifold constraint ``Omega`` chosen by
the intervention family:

    A (Semantic Structure Transport, Sec. 4.2)
        Phi(z) = mapper(z),  Omega(z) = P_subspace @ z
    B.1 (Answer-Anchored Slerp, Sec. 4.3, Eq. 13)
        norm-preserving rotation of the causal hub z_2 toward an anchor t
    B.2 (Answer-Directed Gradient Update, Sec. 4.3, Eq. 14)
        Phi(z) = z - eta * grad L,  Omega(z) = z / ||z||
    C.1 (Weight-Tying Consistent Projection, Sec. 4.4, Eq. 15)
        h' = (1 - alpha) h + alpha * E^T softmax(o / tau)
    C.2 (Energy-Guided Local Descent, Sec. 4.4, Eq. 16)
        h' = h - Proj_{B(0, rho ||h||)}( eta * grad H(h) )
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from .geometry import (
    directional_update,
    manifold_mix,
    slerp,
    slerp_preserve_norm,
    trust_region_clip,
)
from .probes import (
    EnergyMLP,
    Mapper,
    project_to_subspace,
    weight_tying_expected_embedding,
)

# Default latent-chain indices (0-based) used by the paper for K = 6.
TERMINAL_PAIR = (4, 5)   # z5, z6 -- the terminal latent pair (Intervention A)
CAUSAL_HUB = 1           # z2     -- the early causal hub (Intervention B)


def _read(inputs_embeds: torch.Tensor, pos: int) -> torch.Tensor:
    return inputs_embeds[0, pos, :].detach().float()


def _write(inputs_embeds: torch.Tensor, pos: int, value: torch.Tensor) -> None:
    inputs_embeds[0, pos, :] = value.to(inputs_embeds.dtype).to(inputs_embeds.device)


def unified_steer(z: torch.Tensor, phi: Callable[[torch.Tensor], torch.Tensor],
                  omega: Callable[[torch.Tensor], torch.Tensor],
                  alpha: float, lam: float) -> torch.Tensor:
    """Reference implementation of Algorithm 1 for a single latent state."""
    v_star = phi(z)
    z_steer = directional_update(z, v_star, alpha)
    return manifold_mix(z_steer, omega(z_steer), lam)


# --------------------------------------------------------------------------- #
# A. Semantic Structure Transport (Section 4.2)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def semantic_transport(
    inputs_embeds: torch.Tensor,
    latent_positions: List[int],
    mapper: Mapper,
    projector: Optional[torch.Tensor],
    alpha: float = 0.15,
    eta_norm: float = 0.25,
    lam: float = 0.0,
    beta_residual: float = 1.0,
    pair: tuple = TERMINAL_PAIR,
) -> torch.Tensor:
    """Mapper-guided transport of the terminal latent pair (Eq. 12).

    The mean ``m0`` of the terminal latent pair is mapped to its semantic
    destination ``t0 = mapper(m0)``, steered toward it with a norm-preserving
    slerp, renormalized via norm mixing (``eta_norm``), optionally projected
    onto the embedding subspace (``lam``), and finally the instance-specific
    residuals are re-applied (``beta_residual``) to preserve per-token detail.
    """
    a, b = pair
    if max(a, b) >= len(latent_positions):
        return inputs_embeds
    pa, pb = latent_positions[a], latent_positions[b]
    za, zb = _read(inputs_embeds, pa), _read(inputs_embeds, pb)

    m0 = 0.5 * (za + zb)
    t0 = mapper(m0.to(next(mapper.parameters()).device)).detach().cpu().float()

    # Eq. (12): direction via slerp, magnitude via norm mixing.
    direction = slerp(m0, t0, alpha)
    new_norm = (1.0 - eta_norm) * m0.norm() + eta_norm * t0.norm()
    m1 = new_norm * direction

    # Manifold regularization onto the embedding subspace (Algorithm 1, line 4).
    if projector is not None and lam > 0:
        m1 = manifold_mix(m1, project_to_subspace(m1, projector), lam)

    # Re-apply instance-specific residuals around the new mean.
    ra, rb = za - m0, zb - m0
    _write(inputs_embeds, pa, m1 + beta_residual * ra)
    _write(inputs_embeds, pb, m1 + beta_residual * rb)
    return inputs_embeds


# --------------------------------------------------------------------------- #
# B.1 Answer-Anchored Slerp (Section 4.3, Eq. 13)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def causal_anchored_slerp(
    inputs_embeds: torch.Tensor,
    latent_positions: List[int],
    anchor: torch.Tensor,
    alpha: float = 0.10,
    hub_index: int = CAUSAL_HUB,
) -> torch.Tensor:
    """Reorient the causal hub ``z_2`` toward an answer-supporting anchor.

        z'_2 = ||z_2|| * slerp(z_2 / ||z_2||, t / ||t||; alpha)

    ``anchor`` is the boundary/target vector ``t`` (e.g. the final latent thought
    of a top-1 retrieved exemplar; see :mod:`ili.latent_cache`).
    """
    if hub_index >= len(latent_positions) or anchor is None:
        return inputs_embeds
    pos = latent_positions[hub_index]
    z = _read(inputs_embeds, pos)
    _write(inputs_embeds, pos, slerp_preserve_norm(z, anchor.float(), alpha))
    return inputs_embeds


# --------------------------------------------------------------------------- #
# B.2 Answer-Directed Gradient Update (Section 4.3, Eq. 14)
# --------------------------------------------------------------------------- #
def causal_gradient_update(
    model,
    inputs_embeds: torch.Tensor,
    latent_positions: List[int],
    answer_ids: List[int],
    eta: float = 0.20,
    hub_index: int = CAUSAL_HUB,
    eps: float = 1e-6,
) -> torch.Tensor:
    """One normalized, norm-preserving gradient step on the causal hub.

        z'_2 = z_2 - eta * ||z_2|| * grad_{z_2} L / (||grad_{z_2} L|| + eps)

    where ``L`` is the teacher-forced cross-entropy of the correct answer tokens
    appended after the prefix. The step direction maximizes answer likelihood;
    the explicit norm scaling prevents energy explosion (Algorithm 1: Omega is
    the norm projection ``z / ||z||``).
    """
    if hub_index >= len(latent_positions) or len(answer_ids) == 0:
        return inputs_embeds

    model_dtype = next(model.parameters()).dtype
    device = inputs_embeds.device
    pos = latent_positions[hub_index]

    prefix = inputs_embeds.detach().clone().requires_grad_(True)
    ans = torch.tensor([answer_ids], device=device, dtype=torch.long)
    ans_embeds = model.get_input_embeddings()(ans).to(model_dtype)
    full = torch.cat([prefix, ans_embeds], dim=1)

    labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device=device)
    labels[:, -len(answer_ids):] = ans
    loss = model(inputs_embeds=full, labels=labels).loss

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = prefix.grad[0, pos, :].float()

    out = inputs_embeds.detach().clone()
    if torch.isfinite(grad).all() and float(grad.norm()) > 0:
        z = _read(out, pos)
        direction = -grad / (grad.norm() + eps)
        _write(out, pos, z + eta * z.norm() * direction)
    return out


# --------------------------------------------------------------------------- #
# C.1 Weight-Tying Consistent Projection (Section 4.4, Eq. 15)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def geometric_weight_tying(
    model,
    inputs_embeds: torch.Tensor,
    latent_positions: List[int],
    embedding_weight: torch.Tensor,
    alpha: float = 0.20,
    tau: float = 1.0,
    target_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Nudge latent states toward their expected tied-embedding (Eq. 15).

        e_tilde = E^T softmax(o_{l-1} / tau)
        h'_l    = (1 - alpha) h_l + alpha * e_tilde

    A single forward pass over the current prefix supplies the logits ``o_{l-1}``
    that precede each latent slot.
    """
    if alpha <= 0 or len(latent_positions) == 0:
        return inputs_embeds
    logits = model(inputs_embeds=inputs_embeds).logits[0]  # [L, V]
    idxs = target_indices if target_indices is not None else range(len(latent_positions))
    for j in idxs:
        pos = latent_positions[j]
        if pos <= 0:
            continue
        e_tilde = weight_tying_expected_embedding(logits[pos - 1], embedding_weight, tau)
        h = _read(inputs_embeds, pos)
        _write(inputs_embeds, pos, (1.0 - alpha) * h + alpha * e_tilde)
    return inputs_embeds


# --------------------------------------------------------------------------- #
# C.2 Energy-Guided Local Descent (Section 4.4, Eq. 16)
# --------------------------------------------------------------------------- #
def _energy_descent(h: torch.Tensor, energy: EnergyMLP, eta: float,
                    radius_ratio: float, steps: int) -> torch.Tensor:
    with torch.enable_grad():
        x = h.detach().float().clone().requires_grad_(True)
        base_norm = float(x.norm()) + 1e-6
        radius = radius_ratio * base_norm
        for _ in range(max(1, steps)):
            e = energy(x.unsqueeze(0)).sum()
            grad = torch.autograd.grad(e, x)[0]
            step = trust_region_clip(eta * grad, radius)  # Eq. (16) projection
            x = (x - step).detach().requires_grad_(True)
        return x.detach()


def geometric_energy_descent(
    inputs_embeds: torch.Tensor,
    latent_positions: List[int],
    energy: EnergyMLP,
    eta: float = 2e-3,
    radius_ratio: float = 0.25,
    steps: int = 1,
    target_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Trust-region energy descent on latent states (Eq. 16).

    Drives each latent "downhill" on the learned energy landscape H, stabilizing
    the trajectory while a trust region of radius ``rho * ||h||`` bounds the step.
    """
    if energy is None or len(latent_positions) == 0:
        return inputs_embeds
    idxs = target_indices if target_indices is not None else range(len(latent_positions))
    for j in idxs:
        pos = latent_positions[j]
        h = _read(inputs_embeds, pos)
        _write(inputs_embeds, pos, _energy_descent(h, energy, eta, radius_ratio, steps))
    return inputs_embeds
