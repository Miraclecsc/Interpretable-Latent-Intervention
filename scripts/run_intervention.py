#!/usr/bin/env python
"""Run a training-free latent intervention and report accuracy.

Examples
--------
Baseline (write cached latents, no intervention):

    python scripts/run_intervention.py --intervention none \
        --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut.pt \
        --test_path data/gsm8k_test.json --latent_root cache/gsm8k

Answer-Directed Gradient Update (Intervention B.2):

    python scripts/run_intervention.py --intervention gradient \
        --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut.pt \
        --test_path data/gsm8k_test.json --latent_root cache/gsm8k --grad_eta 0.20
"""

import argparse
import json
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ili import InterventionConfig, evaluate, INTERVENTIONS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intervention", choices=INTERVENTIONS, default="none")
    p.add_argument("--model_base", required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--test_path", required=True)
    p.add_argument("--latent_root", required=True)
    p.add_argument("--mapper_path", default="")
    p.add_argument("--energy_path", default="")
    p.add_argument("--anchor_root", default="")
    p.add_argument("--out_json", default="")

    p.add_argument("--num_latent", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--paradigm", choices=["coconut", "codi"], default="coconut")
    p.add_argument("--limit", type=int, default=0)

    # Intervention hyper-parameters (defaults = Table 6 optima).
    p.add_argument("--mapper_alpha", type=float, default=0.15)
    p.add_argument("--mapper_eta_norm", type=float, default=0.25)
    p.add_argument("--mapper_lambda", type=float, default=0.0)
    p.add_argument("--mapper_beta_residual", type=float, default=1.0)
    p.add_argument("--pc_rank", type=int, default=1024)
    p.add_argument("--slerp_alpha", type=float, default=0.10)
    p.add_argument("--grad_eta", type=float, default=0.20)
    p.add_argument("--wt_alpha", type=float, default=0.20)
    p.add_argument("--wt_tau", type=float, default=1.0)
    p.add_argument("--energy_eta", type=float, default=2e-3)
    p.add_argument("--energy_radius_ratio", type=float, default=0.25)
    p.add_argument("--energy_steps", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    valid = {f.name for f in fields(InterventionConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in valid}
    kwargs["use_bf16"] = not args.no_bf16
    cfg = InterventionConfig(**kwargs)

    result = evaluate(cfg)
    print(f"\n=== {result['intervention']} ===")
    print(f"Accuracy = {result['accuracy']:.4f}  (n={result['n']})")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
