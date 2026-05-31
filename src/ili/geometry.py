"""Geometric primitives for latent steering.

This module implements the low-level, norm-aware vector operations that the
training-free interventions in the paper are built on (Section 4, Algorithm 1).
All functions operate on ``float32`` tensors and are device-agnostic.
"""

from __future__ import annotations

import torch


def unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return ``v`` normalized to unit L2 norm."""
    return v / (v.norm() + eps)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1-D tensors."""
    return float((a @ b) / (a.norm() * b.norm() + eps))


def slerp(u: torch.Tensor, v: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    """Spherical linear interpolation between the *directions* of ``u`` and ``v``.

    Returns a unit vector that is rotated ``alpha`` of the way from ``u`` toward
    ``v`` along the geodesic of the unit sphere. The caller is responsible for
    re-applying a norm (the interventions keep the original ``||z||``), which is
    what makes the edit norm-preserving and therefore manifold-friendly.
    """
    un, vn = unit(u, eps), unit(v, eps)
    dot = float(torch.clamp((un * vn).sum(), -1.0, 1.0))
    if dot > 0.9995:
        # Nearly colinear: fall back to LERP to avoid division by sin(theta)->0.
        return unit((1.0 - alpha) * un + alpha * vn, eps)
    theta = torch.acos(torch.tensor(dot))
    s1 = torch.sin((1.0 - alpha) * theta) / torch.sin(theta)
    s2 = torch.sin(alpha * theta) / torch.sin(theta)
    return unit(s1 * un + s2 * vn, eps)


def slerp_preserve_norm(z: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Reorient ``z`` toward ``target`` by ``alpha`` while preserving ``||z||``.

    Implements the norm-preserving answer-anchored edit, e.g. Eq. (13):
        z' = ||z|| * slerp(z/||z||, t/||t||; alpha).
    """
    return z.norm() * slerp(z, target, alpha)


def directional_update(z: torch.Tensor, target: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    """Norm-scaled directional step toward ``target`` (Algorithm 1, lines 2-3).

        d        = target - z
        z_steer  = z + alpha * (d / ||d||) * ||z||
    """
    d = target - z
    return z + alpha * (d / (d.norm() + eps)) * z.norm()


def manifold_mix(z_steer: torch.Tensor, z_constrained: torch.Tensor, lam: float) -> torch.Tensor:
    """Manifold regularization (Algorithm 1, line 4): convex mix with Omega(z)."""
    return (1.0 - lam) * z_steer + lam * z_constrained


def trust_region_clip(delta: torch.Tensor, radius: float) -> torch.Tensor:
    """Project a step ``delta`` into the ball of radius ``radius`` (Eq. 16)."""
    n = float(delta.norm())
    if n > radius and n > 0:
        return delta * (radius / n)
    return delta
