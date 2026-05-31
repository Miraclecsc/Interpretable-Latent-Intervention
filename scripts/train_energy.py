#!/usr/bin/env python
"""Train the monotone energy function H over the latent chain (Section 3.4).

H is a small MLP optimized with a margin ranking loss (Eq. 11) so that energy
strictly decreases at each step along the chain, from a high-energy "unresolved"
state to a low-energy "resolved" state::

    L_rank = E[ max(0, gamma - (H(x_t) - H(x_{t+1}))) ]

Each training chain is the cached latent sequence ``z_1, ..., z_K``. When cached
anchor states are available (``cot_root`` with ``answer_start.pt``) the answer
state is appended as the terminal (lowest-energy) node. The trained H provides
the geometric prior consumed by Intervention C.2 (Energy-Guided Local Descent).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ili.latent_cache import load_cot_targets, load_latent_chain  # noqa: E402
from ili.probes import EnergyMLP  # noqa: E402


def build_chains(latent_root, cot_root, num_latent):
    chains = []
    for case in tqdm(sorted(glob.glob(os.path.join(latent_root, "case_*"))),
                     desc="load chains", dynamic_ncols=True):
        idx = int(os.path.basename(case).split("_")[1])
        chain = load_latent_chain(latent_root, idx, num_latent)
        if chain is None:
            continue
        nodes = [chain[j] for j in range(chain.shape[0])]
        if cot_root:
            tgt = load_cot_targets(cot_root, idx)
            if "answer_start" in tgt:
                nodes.append(tgt["answer_start"])  # terminal resolved state
        chains.append(torch.stack(nodes, dim=0).numpy().astype(np.float32))
    if not chains:
        raise RuntimeError("No latent chains found; check --latent_root.")
    return chains


def rank_monotonicity(model, chains, device):
    model.eval()
    strict, sp = 0, []
    with torch.no_grad():
        for c in chains:
            H = model(torch.tensor(c, device=device)).cpu().numpy().tolist()
            strict += float(all(H[i] > H[i + 1] for i in range(len(H) - 1)))
            order = np.argsort(np.argsort([-h for h in H]))
            ideal = np.arange(len(H))
            sp.append(float(np.corrcoef(order, ideal)[0, 1]) if len(H) > 1 else 1.0)
    return strict / len(chains), float(np.mean(sp))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent_root", required=True)
    ap.add_argument("--cot_root", default="")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--num_latent", type=int, default=6)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chains = build_chains(args.latent_root, args.cot_root, args.num_latent)
    dim = chains[0].shape[1]
    perm = np.random.RandomState(args.seed).permutation(len(chains))
    n_tr = int(round(len(chains) * args.train_ratio))
    tr = [chains[i] for i in perm[:n_tr]]
    te = [chains[i] for i in perm[n_tr:]] or tr

    # Whitening statistics over all training nodes (stored inside H).
    allnodes = np.concatenate(tr, axis=0)
    mu = torch.tensor(allnodes.mean(0), dtype=torch.float32)
    sigma = torch.tensor(allnodes.std(0) + 1e-6, dtype=torch.float32)

    model = EnergyMLP(dim, args.hidden_dim, args.num_layers).to(device)
    model.mu.copy_(mu.to(device)); model.sigma.copy_(sigma.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs = [(c[t], c[t + 1]) for c in tr for t in range(c.shape[0] - 1)]
    pairs = [(torch.tensor(a), torch.tensor(b)) for a, b in pairs]

    best = {"loss": 1e9, "state": None}
    for ep in range(args.epochs):
        model.train()
        order = np.random.permutation(len(pairs))
        losses = []
        for s in range(0, len(order), args.batch_size):
            sub = order[s:s + args.batch_size]
            A = torch.stack([pairs[j][0] for j in sub]).to(device)
            B = torch.stack([pairs[j][1] for j in sub]).to(device)
            # Eq. (11): energy of earlier node should exceed later by the margin.
            loss = torch.clamp(args.margin - (model(A) - model(B)), min=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        tr_loss = float(np.mean(losses)) if losses else 0.0
        strict, sp = rank_monotonicity(model, te, device)
        print(f"[ep {ep + 1:02d}] loss={tr_loss:.4f} strict={strict:.3f} spearman={sp:.3f}")
        if tr_loss < best["loss"]:
            best = {"loss": tr_loss, "state": {k: v.detach().cpu()
                                               for k, v in model.state_dict().items()}}

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(best["state"], args.out_path)
    strict, sp = rank_monotonicity(model, te, device)
    print(f"Saved energy H -> {args.out_path} | strict={strict:.3f} spearman={sp:.3f}")
    with open(os.path.splitext(args.out_path)[0] + "_metrics.json", "w") as f:
        json.dump({"dim": dim, "n_chains": len(chains),
                   "strict_monotonic": strict, "spearman": sp}, f, indent=2)


if __name__ == "__main__":
    main()
