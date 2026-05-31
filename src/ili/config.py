"""Configuration for the decode-time interventions.

Default hyper-parameters are the optimal values reported in Table 6 of the
paper (grid-searched on Qwen3-8B / CoConut, then held fixed across all model
scales and datasets).
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical intervention identifiers (paper Sections 4.2-4.4).
INTERVENTIONS = ("none", "mapper", "slerp", "gradient", "wt_proj", "energy")


@dataclass
class InterventionConfig:
    # --- which intervention to run ---
    intervention: str = "none"        # one of INTERVENTIONS

    # --- paths ---
    model_base: str = ""              # HF id or local path of the backbone
    ckpt_path: str = ""               # trained latent-reasoning checkpoint
    test_path: str = ""               # eval JSON ({question, steps, answer})
    latent_root: str = ""             # cached continuous thoughts (per case)
    mapper_path: str = ""             # trained mapper f_phi (intervention A)
    energy_path: str = ""             # trained energy H (intervention C.2)
    anchor_root: str = ""             # exemplar latents for retrieval (B.1);
                                      # defaults to latent_root when empty

    # --- runtime ---
    num_latent: int = 6               # K
    max_new_tokens: int = 128
    use_bf16: bool = True
    paradigm: str = "coconut"         # "coconut" | "codi" (documentation only)
    limit: int = 0                    # evaluate only the first N examples (0=all)

    # --- A: Semantic Structure Transport (Sec. 4.2 / Table 6) ---
    mapper_alpha: float = 0.15        # slerp factor
    mapper_eta_norm: float = 0.25     # norm-mixing weight
    mapper_lambda: float = 0.0        # embedding-subspace mix weight
    mapper_beta_residual: float = 1.0 # residual re-application scale
    pc_rank: int = 1024               # PCs kept for the subspace projector

    # --- B.1: Answer-Anchored Slerp (Sec. 4.3, Eq. 13 / Table 6) ---
    slerp_alpha: float = 0.10

    # --- B.2: Answer-Directed Gradient Update (Sec. 4.3, Eq. 14 / Table 6) ---
    grad_eta: float = 0.20

    # --- C.1: Weight-Tying Consistent Projection (Sec. 4.4, Eq. 15 / Table 6) ---
    wt_alpha: float = 0.20
    wt_tau: float = 1.0

    # --- C.2: Energy-Guided Local Descent (Sec. 4.4, Eq. 16 / Table 6) ---
    energy_eta: float = 2e-3
    energy_radius_ratio: float = 0.25
    energy_steps: int = 1

    def validate(self) -> None:
        if self.intervention not in INTERVENTIONS:
            raise ValueError(
                f"intervention must be one of {INTERVENTIONS}, got {self.intervention!r}")
