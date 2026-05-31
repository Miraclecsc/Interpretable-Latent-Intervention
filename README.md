# Interpretable Latent Intervention

Official repository for **"Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention"**, accepted to **ACL 2026 Main**.

This project studies how latent reasoning unfolds inside continuous hidden states, and how those states can be interpreted and intervened on at decode time. Instead of treating latent thoughts as opaque vectors, we analyze their structure, semantic content, and causal role, then use the resulting observations to design training-free interventions that improve reasoning.

<p align="center">
  <img src="assets/coconut.pdf.png" width="88%" alt="Explicit reasoning versus latent reasoning">
</p>

## Overview

Latent reasoning allows language models to perform multi-step inference through continuous internal states rather than explicit Chain-of-Thought tokens. This can reduce decoding cost, but it also makes the reasoning process harder to inspect or control.

We approach this problem in two stages:

1. **Interpretation.** We probe latent thought vectors using structural alignment, linear recoverability, lexical probing, and causal interventions. The results show that latent vectors encode compressed semantic representations of reasoning steps, with early latent states playing a particularly important causal role.
2. **Intervention.** We convert these interpretability findings into training-free, decode-time interventions that steer latent reasoning trajectories without updating model parameters.

<p align="center">
  <img src="assets/alignment.pdf.png" width="78%" alt="Alignment between latent thoughts and explicit reasoning representations">
</p>

## Highlights

- Latent thought vectors exhibit strong geometric alignment with explicit reasoning states.
- Reasoning-step representations are linearly recoverable from latent vectors.
- Early latent states act as causal hubs for the final answer.
- Simple decode-time interventions can improve reasoning accuracy without parameter updates.
- Experiments cover multiple model scales and reasoning domains, including mathematical reasoning and commonsense reasoning benchmarks.

<p align="center">
  <img src="assets/intervention_gains.pdf.png" width="78%" alt="Decode-time intervention gains">
</p>

## Repository Status

The codebase is being organized for public release.

Planned components:

- latent-reasoning inference utilities;
- probing and alignment scripts;
- causal intervention tools;
- decode-time intervention implementations;
- experiment configs and evaluation scripts;
- processed figure-generation scripts.

The current repository is a landing page for the paper and will be updated with code and checkpoints as they are prepared for release.

## Citation

```bibtex
@inproceedings{chang2026latentintervention,
  title = {Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention},
  author = {Chang, Shuochen and Bai, Tong and Zhang, Xiaofeng and Ma, Qianli and Liu, Qingyang and Liao, Zhaohe and Miao, Yibo and Niu, Li},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year = {2026}
}
```

## License

The license will be finalized with the code release.
