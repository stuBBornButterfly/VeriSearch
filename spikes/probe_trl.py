"""Spike probe: which multi-turn hooks does the installed TRL expose?

Prints facts about the installed version only. Draws no conclusions.

Usage:
    .venv/bin/python spikes/probe_trl.py
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as md
import inspect
import re
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

DEPS = ("trl", "torch", "transformers", "vllm", "peft", "accelerate")
CONFIG_PATTERN = re.compile(r"rollout|env|tool|mask|generat|vllm|turn", re.IGNORECASE)
SOURCE_TERMS = (
    "rollout_func",
    "environment",
    "tool_mask",
    "env_mask",
    "completion_mask",
    "multi_turn",
    "max_turns",
)
MAX_HITS = 5


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("Versions")
    for dist in DEPS:
        print(f"  {dist:<13}{_version(dist)}")

    print("\nGRPOTrainer.__init__ parameters")
    for name, param in inspect.signature(GRPOTrainer.__init__).parameters.items():
        if name != "self":
            print(f"  {name}: {param.annotation}")

    print("\nGRPOConfig fields matching rollout/env/tool/mask/generat/vllm/turn")
    for field in dataclasses.fields(GRPOConfig):
        if CONFIG_PATTERN.search(field.name):
            default = "<factory>" if field.default is dataclasses.MISSING else repr(field.default)
            print(f"  {field.name} = {default}")

    source = Path(inspect.getfile(GRPOTrainer))
    lines = source.read_text(encoding="utf-8").splitlines()
    print(f"\nTerm hits in {source}")
    for term in SOURCE_TERMS:
        hits = [i for i, line in enumerate(lines, start=1) if term in line]
        shown = ", ".join(str(i) for i in hits[:MAX_HITS])
        print(f"  {term:<16}{len(hits):>4} hits  lines {shown or '-'}")


if __name__ == "__main__":
    main()
