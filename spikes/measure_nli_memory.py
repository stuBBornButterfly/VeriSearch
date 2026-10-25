"""Peak GPU memory and latency of a DeBERTa-v3-base NLI scorer, for finding B on #31.

Stand-in for #7's scorer: `cross-encoder/nli-deberta-v3-base` has the same
backbone and size as base.yaml's `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
(DeBERTa-v3-base, 184M parameters, 3-way MNLI head), and is already in the local
cache. fp32, as #7 loads it (no dtype set).

Worst case per GRPO step: 8 rollouts x up to 3 claims x up to 12 passages
(4 searches x top_k 3) = 288 (passage, claim) pairs, each truncated to 512
tokens. Real passages are shorter, so this over-states both numbers.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "cross-encoder/nli-deberta-v3-base"
PAIRS_PER_STEP = 8 * 3 * 12

tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(NAME, local_files_only=True)
model = model.to("cuda").eval()
torch.cuda.synchronize()
print(f"model         {NAME}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
print(f"weights       {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

passage = " ".join(["Babbage designed the Analytical Engine in London."] * 60)
claim = "Ada Lovelace wrote the first published notes on the Analytical Engine."

for batch in (16, 32, 64):
    encoded = tokenizer(
        [passage] * batch,
        [claim] * batch,
        truncation="only_first",
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():  # warm-up
        model(**encoded)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        model(**encoded).logits.softmax(-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_step = elapsed * PAIRS_PER_STEP / batch
    print(
        f"batch {batch:3d}     seq {encoded['input_ids'].shape[1]}  "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  "
        f"{elapsed * 1000:.0f} ms/batch  ~{per_step:.1f} s for {PAIRS_PER_STEP} pairs"
    )
