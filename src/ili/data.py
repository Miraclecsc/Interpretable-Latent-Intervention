"""Dataset construction for the continuous-thought paradigm.

We expect a JSON list of examples, each with at least::

    {"question": "...", "steps": ["...", "..."], "answer": "18"}

For evaluation we build, per example, the *question-latent prefix*:

    <question tokens> <|start-latent|> <|latent|> x K <|end-latent|>

The K reserved ``<|latent|>`` slots are where cached continuous thoughts are
written before decoding (see :mod:`ili.paradigms`). ``steps`` are only used for
the interpretability probes (mapper / energy training), not for evaluation.
"""

from __future__ import annotations

import json
from typing import Dict, List

DEFAULT_NUM_LATENT = 6  # K = 6 throughout the paper.


def load_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gt_answers(path: str) -> List[str]:
    """Ground-truth answers, normalized to match :func:`extract_prediction`."""
    return [d["answer"].replace(",", "").strip() for d in load_examples(path)]


def build_question_latent_items(
    examples: List[Dict],
    tokenizer,
    start_id: int,
    latent_id: int,
    end_id: int,
    num_latent: int = DEFAULT_NUM_LATENT,
    no_special_marker: bool = False,
) -> List[Dict]:
    """Tokenize each example into a question-latent generation prefix.

    Returns a list of ``{"idx": int, "input_ids": List[int]}`` ready to be turned
    into ``inputs_embeds``. The latent slots are placeholder ``latent_id`` tokens
    whose embeddings are overwritten with cached continuous thoughts at runtime.
    """
    items: List[Dict] = []
    for idx, ex in enumerate(examples):
        q_ids = tokenizer.encode(ex["question"] + "\n", add_special_tokens=True)
        ids = list(q_ids)
        if not no_special_marker:
            ids.append(start_id)
        ids.extend([latent_id] * num_latent)
        if not no_special_marker:
            ids.append(end_id)
        items.append({"idx": idx, "input_ids": ids})
    return items
