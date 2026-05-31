#!/usr/bin/env python
"""Train the latent-to-CoT mapper f_phi (Section 3.2, Appendix B.1).

The mapper tests *linear recoverability*: it learns to reconstruct an explicit
CoT step representation ``c_t`` from the corresponding latent surrogate
``z_bar_t`` (the local average of contiguous latents). Following the paper we
optimize a cosine objective (Eq. 7), grouping the standard step alignment::

    (z1 + z2) / 2  -> step_end_01
    (z3 + z4) / 2  -> step_end_02
    (z5 + z6) / 2  -> answer_start

Inputs are the cached latents (``--latent_root``) and the cached CoT step
hidden states (``--cot_root``). The trained mapper is consumed by Intervention A.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ili.latent_cache import load_cot_targets, load_latent_chain  # noqa: E402
from ili.probes import Mapper  # noqa: E402

PAIR_TO_TARGET = {(0, 1): "step_end_01", (2, 3): "step_end_02", (4, 5): "answer_start"}


def collect_pairs(latent_root: str, cot_root: str, num_latent: int):
    samples = []
    cases = sorted(glob.glob(os.path.join(latent_root, "case_*")))
    for case in tqdm(cases, desc="scan cases", dynamic_ncols=True):
        idx = int(os.path.basename(case).split("_")[1])
        chain = load_latent_chain(latent_root, idx, num_latent)
        if chain is None:
            continue
        targets = load_cot_targets(cot_root, idx)
        for (a, b), name in PAIR_TO_TARGET.items():
            if name in targets and b < chain.shape[0]:
                x = 0.5 * (chain[a] + chain[b])
                samples.append((x.numpy(), targets[name].numpy(), name, idx))
    if not samples:
        raise RuntimeError("No (latent, CoT) pairs collected; check the cache paths.")
    return samples


class PairDS(Dataset):
    def __init__(self, items):
        self.X = np.stack([s[0] for s in items]).astype(np.float32)
        self.Y = np.stack([s[1] for s in items]).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def split_by_case(samples, val_frac, test_frac, seed):
    cases = sorted({s[3] for s in samples})
    random.Random(seed).shuffle(cases)
    n = len(cases)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test, val = set(cases[:n_test]), set(cases[n_test:n_test + n_val])
    train = set(cases[n_test + n_val:])
    pick = lambda S: [s for s in samples if s[3] in S]
    return pick(train), pick(val), pick(test)


@torch.no_grad()
def cosine_delta(model, loader, device):
    model.eval()
    after, base = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = F.normalize(model(x), dim=-1)
        yn = F.normalize(y, dim=-1)
        after += (p * yn).sum(-1).cpu().tolist()
        base += (F.normalize(x, dim=-1) * yn).sum(-1).cpu().tolist()
    return float(np.mean(after)), float(np.mean(after)) - float(np.mean(base))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent_root", required=True)
    ap.add_argument("--cot_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_latent", type=int, default=6)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=2, choices=[1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    samples = collect_pairs(args.latent_root, args.cot_root, args.num_latent)
    dim = samples[0][0].shape[0]
    tr, va, te = split_by_case(samples, args.val_frac, args.test_frac, args.seed)
    tl = DataLoader(PairDS(tr), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(PairDS(va), batch_size=args.batch_size)
    el = DataLoader(PairDS(te), batch_size=args.batch_size)

    model = Mapper(dim, args.hidden_dim, args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_delta, best_path = -1e9, os.path.join(args.out_dir, "mapper.pt")
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = (1.0 - F.cosine_similarity(model(x), y, dim=-1)).mean()  # Eq. (7)
            opt.zero_grad(); loss.backward(); opt.step()
        cos, delta = cosine_delta(model, vl, device)
        print(f"[ep {ep + 1:02d}] val cos={cos:.4f} delta={delta:+.4f}")
        if delta > best_delta:
            best_delta = delta
            torch.save({"model": model.state_dict(),
                        "meta": {"d": dim, "hidden_dim": args.hidden_dim,
                                 "num_layers": args.num_layers}}, best_path)

    model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    te_cos, te_delta = cosine_delta(model, el, device)
    report = {"dim": dim, "n_pairs": len(samples), "test_cosine": te_cos,
              "test_delta_vs_identity": te_delta, "checkpoint": best_path}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Test cosine={te_cos:.4f} (delta {te_delta:+.4f}) -> {best_path}")


if __name__ == "__main__":
    main()
