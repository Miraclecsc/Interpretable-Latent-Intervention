# Interpretable Latent Intervention

> Official repository for **"Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention"**, accepted to **ACL 2026 Main**.

📄 **Paper:** [`paper/Interpretable_Latent_Intervention_ACL2026.pdf`](paper/Interpretable_Latent_Intervention_ACL2026.pdf)

## 📋 Overview

Latent reasoning enables language models to perform multi-step inference through continuous hidden states rather than explicit Chain-of-Thought tokens. While this paradigm can reduce decoding cost, it also makes the reasoning process harder to inspect, understand, and control.

This project investigates how latent reasoning unfolds inside continuous representations. Instead of treating latent thoughts as opaque vectors, we analyze their geometric structure, semantic content, and causal role in reasoning. Based on these observations, we further design training-free decode-time interventions that steer latent reasoning trajectories without updating model parameters.

<p align="center">
  <img src="assets/coconut.pdf.png" width="88%" alt="Explicit reasoning versus latent reasoning">
</p>

## 🔍 Method

We study latent reasoning from two complementary perspectives:

### Interpretation

We probe latent thought vectors through structural alignment, linear recoverability, lexical probing, and causal intervention analysis. Our findings suggest that latent vectors encode compressed semantic representations of intermediate reasoning steps, with early latent states playing a particularly important causal role in determining the final answer.

<p align="center">
  <img src="assets/alignment.pdf.png" width="78%" alt="Alignment between latent thoughts and explicit reasoning representations">
</p>

### Intervention

We convert the interpretability findings into training-free decode-time interventions. These interventions operate directly on latent reasoning states and steer the model's internal reasoning trajectory without modifying model parameters.

<p align="center">
  <img src="assets/intervention_gains.pdf.png" width="78%" alt="Decode-time intervention gains">
</p>

## ✨ Highlights

- Latent thought vectors exhibit strong geometric alignment with explicit reasoning states.
- Reasoning-step representations are linearly recoverable from latent vectors.
- Early latent states act as causal hubs for the final answer.
- Training-free decode-time interventions can improve reasoning accuracy without parameter updates.
- Experiments cover multiple model scales and reasoning domains, including mathematical and commonsense reasoning benchmarks.


## 🗂️ Repository Structure

```
src/ili/
  geometry.py        # slerp, norm-preserving directional update, trust-region clip (Alg. 1 primitives)
  probes.py          # Mapper f_phi, EnergyMLP H, embedding-subspace projector, weight-tying expectation
  interventions.py   # the five decode-time interventions (A, B.1, B.2, C.1, C.2)
  paradigms.py       # write cached latents into the prefix (CoConut / CODI)
  latent_cache.py    # load cached thoughts / CoT hiddens, exemplar retrieval for the anchor
  model_utils.py     # load backbone + latent special tokens + merge trained checkpoint
  decoding.py        # greedy decoding from inputs_embeds, answer parsing
  data.py            # build the question-latent prefix from a JSON dataset
  config.py / runner.py  # configuration + end-to-end evaluation
scripts/
  run_intervention.py    # main inference entry point
  train_mapper.py        # train f_phi (§3.2) for Intervention A
  train_energy.py        # train H (§3.4) for Intervention C.2
configs/default.yaml     # reference hyper-parameters (Table 6 optima)
data/sample_gsm8k.json   # example dataset format
paper/                   # the ACL 2026 paper (PDF)
```

## 🚀 Quick Start

The method is **training-free at inference**: it edits *cached* continuous
thoughts right before the answer is decoded. The workflow below takes you from a
raw dataset to intervention accuracy in five steps.

### Step 0 — Install

```bash
pip install -r requirements.txt    # torch, transformers, numpy, tqdm
pip install -e .                   # optional: exposes the `ili` package
```

You will also need a trained latent-reasoning backbone, i.e. a **CoConut** or
**CODI** checkpoint (`--ckpt_path`) on top of a base model such as
`Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B`, or `meta-llama/Llama-3.2-3B`. We
follow the official CoConut/CODI training recipes (K = 6 latent slots.

### Step 1 — Prepare your dataset

A JSON list; `question`/`answer` are required, `steps` are only used when you
train the probes (Step 3). See [`data/sample_gsm8k.json`](data/sample_gsm8k.json).

```json
[{"question": "...", "steps": ["step 1 ...", "step 2 ..."], "answer": "72"}]
```

The predicted answer is parsed as the text after the last `#`, so keep your
answer convention consistent (e.g. the model emits `... # 72`).

### Step 2 — Provide the cached latents

Because the approach is training-free, the latent (continuous-thought) pass is
run **once** with your CoConut/CODI model and saved to disk; every intervention
then reads from this cache. Expected layout (one folder per example, indices
aligned with the dataset order):

```
latent_root/
  case_00000/ latent_01.pt  latent_02.pt  ...  latent_06.pt   # thoughts z_1..z_K, each a [d] tensor
  case_00001/ ...
```

> If a `case_XXXXX/` folder is missing, that example simply falls back to plain
> (un-intervened) decoding, so partial caches are fine.

For the probe-based interventions (A and C.2) you additionally need the explicit
CoT step-boundary hidden states, captured from the same model:

```
cot_root/
  case_00000/ step_end_01.pt  step_end_02.pt  answer_start.pt
```

### Step 3 — (Only for A / C.2) Train the small probes

The mapper (Intervention A) and the energy function (Intervention C.2) are the
**only** learned components — both are tiny networks fit on the cached vectors,
not the LLM. Skip this step if you only use `gradient`, `slerp`, or `wt_proj`.

```bash
# Linear mapper f_phi : latent surrogate -> CoT step representation (cosine loss, Eq. 7)
python scripts/train_mapper.py \
  --latent_root cache/gsm8k --cot_root cache/gsm8k_cot --out_dir ckpt/mapper

# Monotone energy H over the latent chain (margin ranking loss, Eq. 11)
python scripts/train_energy.py \
  --latent_root cache/gsm8k --cot_root cache/gsm8k_cot --out_path ckpt/energy.pt
```

### Step 4 — Run the baseline

Always sanity-check the baseline first (cached latents written, no edit):

```bash
python scripts/run_intervention.py --intervention none \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k
```

> Tip: add `--limit 50` to smoke-test on the first 50 examples, and `--no_bf16`
> if your hardware does not support bfloat16.

### Step 5 — Apply an intervention

Pick a `--intervention` from the table above. Defaults already match the
Table 6 optima, so the flags below are only shown for clarity.

```bash
# B.2  Answer-Directed Gradient Update  (best in-domain; needs only the answer)
python scripts/run_intervention.py --intervention gradient --grad_eta 0.20 \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k

# B.1  Answer-Anchored Slerp  (retrieves an exemplar anchor from --latent_root)
python scripts/run_intervention.py --intervention slerp --slerp_alpha 0.10 \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k

# A.   Semantic Structure Transport  (needs the trained mapper from Step 3)
python scripts/run_intervention.py --intervention mapper --mapper_path ckpt/mapper/mapper.pt \
  --mapper_alpha 0.15 --mapper_eta_norm 0.25 --mapper_lambda 0.5 \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k

# C.1  Weight-Tying Consistent Projection  (no extra artifacts)
python scripts/run_intervention.py --intervention wt_proj --wt_alpha 0.20 --wt_tau 1.0 \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k

# C.2  Energy-Guided Local Descent  (needs the trained energy H from Step 3)
python scripts/run_intervention.py --intervention energy --energy_path ckpt/energy.pt \
  --energy_eta 0.002 --energy_radius_ratio 0.25 \
  --model_base Qwen/Qwen3-8B --ckpt_path ckpt/coconut_qwen3.pt \
  --test_path data/gsm8k_test.json --latent_root cache/gsm8k
```

Add `--out_json results/gradient.json` to dump per-example predictions alongside
the printed accuracy.

### Which intervention should I use?

- **No extra artifacts needed:** `gradient` (B.2), `slerp` (B.1), `wt_proj` (C.1).
- **Strongest in-domain (GSM8K):** `gradient` (B.2).
- **Most robust out-of-domain (GSM-Hard / SVAMP):** `mapper` (A).
- **Need a trained probe first (Step 3):** `mapper` (A) and `energy` (C.2).

All hyper-parameters are exposed as CLI flags; the defaults are transferable
(grid-searched once on Qwen3-8B/CoConut and kept fixed across every model and
dataset in the paper — see Table 6).

## 🐍 Use as a Library

```python
from ili import InterventionConfig, evaluate

cfg = InterventionConfig(
    intervention="gradient",
    model_base="Qwen/Qwen3-8B",
    ckpt_path="ckpt/coconut_qwen3.pt",
    test_path="data/gsm8k_test.json",
    latent_root="cache/gsm8k",
)
result = evaluate(cfg)
print(result["accuracy"], result["n"])
```

The individual operators are also importable directly from
`ili.interventions` (e.g. `semantic_transport`, `causal_gradient_update`,
`geometric_energy_descent`) and act on a prefilled `inputs_embeds` prefix.

## 📄 Citation

```bibtex
@inproceedings{chang2026latent,
  title     = {Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention},
  author    = {Chang, Shuochen and Bai, Tong and Zhang, Xiaofeng and Ma, Qianli and Liu, Qingyang and Liao, Zhaohe and Miao, Yibo and Niu, Li},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```
