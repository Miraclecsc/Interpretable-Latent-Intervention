"""End-to-end evaluation of a single intervention over a dataset.

Pipeline per example:
  1. build the question-latent prefix and embed it;
  2. write the cached continuous thoughts onto the reserved latent slots
     (this reproduces the un-intervened baseline);
  3. apply the selected decode-time intervention (Sections 4.2-4.4);
  4. greedy-decode the answer and compare with ground truth.

If an example has no cached latents, it falls back to plain decoding so that the
evaluation set size stays fixed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from . import interventions as itv
from .config import InterventionConfig
from .data import build_question_latent_items, extract_gt_answers, load_examples
from .decoding import decode_and_predict
from .latent_cache import AnchorRetriever, load_latent_chain
from .model_utils import load_model
from .paradigms import build_prefix, prompt_query_hidden, write_latents
from .probes import EnergyMLP, Mapper, build_embedding_subspace_projector


class _Resources:
    """Lazily-loaded probes / projectors required by the chosen intervention."""

    def __init__(self, loaded, cfg: InterventionConfig):
        self.mapper: Optional[Mapper] = None
        self.projector: Optional[torch.Tensor] = None
        self.energy: Optional[EnergyMLP] = None
        self.retriever: Optional[AnchorRetriever] = None

        dim = loaded.model.get_input_embeddings().weight.shape[1]

        if cfg.intervention == "mapper":
            if not cfg.mapper_path:
                raise ValueError("intervention='mapper' requires --mapper_path")
            self.mapper = Mapper.from_checkpoint(cfg.mapper_path, dim, device=loaded.device)
            if cfg.mapper_lambda > 0:
                self.projector = build_embedding_subspace_projector(
                    loaded.model.get_input_embeddings().weight, cfg.pc_rank)

        elif cfg.intervention == "energy":
            if not cfg.energy_path:
                raise ValueError("intervention='energy' requires --energy_path")
            self.energy = EnergyMLP.from_checkpoint(cfg.energy_path, dim, device=loaded.device)


def _build_answer_ids(tokenizer, gt_answer: str) -> List[int]:
    """Teacher-forcing target ' # <answer>' for the gradient update (B.2)."""
    return tokenizer(" # " + gt_answer, add_special_tokens=False).input_ids


def _apply_intervention(cfg, loaded, res, inputs_embeds, positions, case_idx, gt_answer):
    name = cfg.intervention
    if name == "none" or len(positions) == 0:
        return inputs_embeds

    if name == "mapper":
        return itv.semantic_transport(
            inputs_embeds, positions, res.mapper, res.projector,
            alpha=cfg.mapper_alpha, eta_norm=cfg.mapper_eta_norm,
            lam=cfg.mapper_lambda, beta_residual=cfg.mapper_beta_residual)

    if name == "slerp":
        query = prompt_query_hidden(loaded, inputs_embeds)
        anchor = res.retriever.retrieve(query)
        return itv.causal_anchored_slerp(inputs_embeds, positions, anchor, alpha=cfg.slerp_alpha)

    if name == "gradient":
        answer_ids = _build_answer_ids(loaded.tokenizer, gt_answer)
        return itv.causal_gradient_update(
            loaded.model, inputs_embeds, positions, answer_ids, eta=cfg.grad_eta)

    if name == "wt_proj":
        return itv.geometric_weight_tying(
            loaded.model, inputs_embeds, positions,
            loaded.model.get_input_embeddings().weight,
            alpha=cfg.wt_alpha, tau=cfg.wt_tau)

    if name == "energy":
        return itv.geometric_energy_descent(
            inputs_embeds, positions, res.energy,
            eta=cfg.energy_eta, radius_ratio=cfg.energy_radius_ratio, steps=cfg.energy_steps)

    raise ValueError(f"Unknown intervention {name!r}")


def evaluate(cfg: InterventionConfig, loaded=None) -> Dict:
    """Run the configured intervention and return an accuracy summary."""
    cfg.validate()
    loaded = loaded or load_model(cfg.model_base, cfg.ckpt_path,
                                  use_bf16=cfg.use_bf16)

    examples = load_examples(cfg.test_path)
    gts = extract_gt_answers(cfg.test_path)
    items = build_question_latent_items(
        examples, loaded.tokenizer, loaded.start_id, loaded.latent_id,
        loaded.end_id, num_latent=cfg.num_latent)
    if cfg.limit > 0:
        items = items[: cfg.limit]

    res = _Resources(loaded, cfg)
    if cfg.intervention == "slerp":
        # Build the exemplar bank lazily (defaults to the eval cache itself).
        root = cfg.anchor_root or cfg.latent_root
        res.retriever = AnchorRetriever.from_cache(
            root, [it["idx"] for it in items], num_latent=cfg.num_latent)

    correct, total, records = 0, 0, []
    pbar = tqdm(items, dynamic_ncols=True, desc=f"[{cfg.intervention}]")
    for item in pbar:
        idx = item["idx"]
        inputs_embeds, positions = build_prefix(loaded, item["input_ids"])

        chain = load_latent_chain(cfg.latent_root, idx, cfg.num_latent)
        if chain is not None and positions:
            inputs_embeds = write_latents(inputs_embeds, positions, chain)
            inputs_embeds = _apply_intervention(
                cfg, loaded, res, inputs_embeds, positions, idx, gts[idx])

        _, pred = decode_and_predict(loaded, item["input_ids"], inputs_embeds, cfg.max_new_tokens)
        ok = (idx < len(gts) and pred == gts[idx])
        correct += int(ok)
        total += 1
        records.append({"idx": idx, "pred": pred, "gt": gts[idx], "correct": ok})
        pbar.set_postfix(acc=f"{correct / max(1, total):.4f}")

    return {
        "intervention": cfg.intervention,
        "accuracy": correct / max(1, total),
        "n": total,
        "records": records,
    }
