"""Where agent personas come from.

Order of preference:

1. ``mosaic_path`` in the config (or ``SPM_MOSAIC_PATH``) -- a MOSAIC checkout of
   your own. A relative path is resolved against the repository, not the working
   directory.
2. ``MOSAIC/`` sitting beside this code, which needs no configuration at all.
3. ``persona_path`` relative to this repository, if you have copied a bank in.
4. The built-in fallback below.

The fallback exists so a fresh clone runs and the tests pass without any external
checkout. It is **not** a substitute for the MOSAIC bank in a reported experiment:
twelve hand-written sketches have neither the demographic coverage nor the
provenance, and heterogeneity across agents is exactly what the responsiveness
detector feeds on. Use it to check the pipeline, not to produce Section VIII.
``runner`` records which source was used in ``run.json`` so a run can never be
mistaken for the other kind.
"""
from __future__ import annotations

import json
import os

FALLBACK = [
    "You are a 34-year-old maintenance technician in a small basin settlement. You "
    "trust instruments over opinions and you like to see a number before you agree.",
    "You are a 61-year-old retired schoolteacher. You are careful, you dislike being "
    "rushed to a conclusion, and you tend to wait for several accounts to line up.",
    "You are a 27-year-old software contractor who moved here two years ago. You are "
    "comfortable with probabilities and impatient with vague answers.",
    "You are a 45-year-old smallholder. You have been burned by optimistic claims "
    "about equipment before, so you start sceptical and need repeated evidence.",
    "You are a 52-year-old clinic nurse. You weigh what people around you report "
    "heavily, because in your work consensus usually tracks reality.",
    "You are a 23-year-old apprentice engineer. You are curious, quick to update, "
    "and sometimes change your mind faster than the evidence warrants.",
    "You are a 38-year-old logistics coordinator. You think in trends rather than "
    "single readings and you want to know which way things are moving.",
    "You are a 70-year-old former plant supervisor. You are stubborn once you have "
    "formed a view and you distrust sudden shifts in what everyone is saying.",
    "You are a 31-year-old freelance journalist. You look for who benefits from a "
    "claim and you notice when several people repeat the same line.",
    "You are a 48-year-old accountant. You are conservative with confidence, rarely "
    "going far from the middle unless the evidence is strong.",
    "You are a 29-year-old hydrology graduate. You know the technical background and "
    "you weight your own measurement more than your neighbours' reports.",
    "You are a 56-year-old shopkeeper. You hear a lot of talk across the counter and "
    "you have a good sense of when opinion is running ahead of the facts.",
]


def load(persona_path: str, mosaic_path: str, repo_root: str) -> tuple[list[dict], str]:
    """Return (personas, source_label)."""
    candidates = []
    if mosaic_path:
        root = os.path.expanduser(mosaic_path)
        if not os.path.isabs(root):
            # Relative to the repository, not the working directory, so the same
            # config works no matter where you launch from.
            root = os.path.join(repo_root, root)
        candidates.append((os.path.join(root, persona_path), "mosaic"))
    # A checkout sitting beside the code needs no configuration at all.
    candidates.append((os.path.join(repo_root, "MOSAIC", persona_path), "adjacent"))
    candidates.append((persona_path if os.path.isabs(persona_path)
                       else os.path.join(repo_root, persona_path), "local"))

    for path, label in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if rows:
                return rows, f"{label}:{path}"

    return [{"persona": p} for p in FALLBACK], "fallback:builtin"
