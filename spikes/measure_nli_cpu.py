"""CPU latency of the same DeBERTa-v3-base NLI stand-in, at realistic sizes.

A realistic GRPO step: 8 rollouts x 1 claim (HotpotQA answers are short) x up to
6 passages (2 searches x top_k 3), each passage ~150 words. Worst case is 288
pairs at 512 tokens (see measure_nli_memory.py).
"""

from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(NAME, local_files_only=True).eval()
print(f"cpu threads   {torch.get_num_threads()} of {os.cpu_count()}")

claim = "Ada Lovelace wrote the first published notes on the Analytical Engine."
for words, pairs in ((150, 48), (150, 96), (400, 288)):
    passage = " ".join(["Babbage designed the engine."] * (words // 4))
    encoded = tokenizer(
        [passage] * 16,
        [claim] * 16,
        truncation="only_first",
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        model(**encoded)
        start = time.perf_counter()
        model(**encoded).logits.softmax(-1)
        elapsed = time.perf_counter() - start
    print(
        f"seq {encoded['input_ids'].shape[1]:3d}  {elapsed / 16 * 1000:.0f} ms/pair  "
        f"~{elapsed / 16 * pairs:.1f} s for {pairs} pairs"
    )
