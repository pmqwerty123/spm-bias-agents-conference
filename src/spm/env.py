"""`.env` support for the machine-local settings: which model, where, and with what key.

Two rules, both deliberate:

* **A real environment variable always beats `.env`.** Setting `SPM_ENGINE` in your
  shell for one run must not be silently overridden by a file you edited last week.
* **`.env` only carries provider settings** -- engine, base URL, key, structured
  mode, temperature, persona checkout. Experiment parameters (topology, attack,
  defense, N, T, seed) stay in `configs/spm_experiment.json`, because those belong
  to the experiment and have to be reproducible by anyone. Nothing that changes a
  result lives in an untracked file.

Precedence ends up: `--set` on the command line > environment > `.env` > the JSON
config. Whatever it resolves to is written into `run.json`, so a finished run always
records the values it actually used rather than the ones the config file shows.

No dependency: this is a small parser rather than python-dotenv, which keeps the
requirements at four packages.
"""
from __future__ import annotations

import os

#: env var -> dotted path in the JSON config
PROVIDER_KEYS = {
    "SPM_ENGINE": "engine",
    "SPM_BASE_URL": "base_url",
    "SPM_STRUCTURED": "structured",
    "SPM_TEMPERATURE": "temperature",
    "SPM_MAX_TOKENS": "max_tokens",
    "SPM_REASONING_EFFORT": "reasoning_effort",
    "SPM_CONCURRENCY": "concurrency",
    "SPM_RPM": "rpm",
    "SPM_MOSAIC_PATH": "mosaic_path",
    "SPM_OUT_DIR": "out_dir",
}

#: parsed as numbers rather than strings when they come from the environment
_NUMERIC = {"temperature"}
_INTEGER = {"max_tokens", "concurrency", "rpm"}


def parse(text: str) -> dict[str, str]:
    """Parse .env text. Supports comments, `export`, and quoted values."""
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]                      # quoted: keep it verbatim
        else:
            value = value.split(" #", 1)[0].rstrip()  # unquoted: strip a trailing comment
        out[key] = value
    return out


def load(path: str | None = None, search_from: str | None = None) -> dict[str, str]:
    """Read a .env into os.environ without clobbering variables already set.

    Looks at `path` if given, otherwise `.env` in the working directory and then in
    `search_from` (the repository root). Missing files are not an error.
    """
    candidates = []
    if path:
        candidates.append(path)
    else:
        candidates.append(os.path.join(os.getcwd(), ".env"))
        if search_from:
            candidates.append(os.path.join(search_from, ".env"))

    loaded: dict[str, str] = {}
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        with open(candidate, encoding="utf-8") as f:
            values = parse(f.read())
        for key, value in values.items():
            loaded.setdefault(key, value)
            if key not in os.environ:          # a real env var wins
                os.environ[key] = value
        break                                   # first .env found wins
    return loaded


def apply(raw: dict) -> dict:
    """Overlay provider settings from the environment onto a parsed JSON config.

    Mutates and returns `raw`. Call this after reading the JSON and *before*
    applying any `--set` overrides, so the command line still has the last word.
    """
    for var, field in PROVIDER_KEYS.items():
        value = os.environ.get(var)
        if value is None or value == "":
            continue
        if field in _NUMERIC:
            raw[field] = float(value)
        elif field in _INTEGER:
            raw[field] = int(value)
        else:
            raw[field] = value
    return raw


def describe() -> str:
    """One line naming which provider settings the environment is currently supplying."""
    active = [v for v in PROVIDER_KEYS if os.environ.get(v)]
    if os.environ.get("SPM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        active.append("SPM_API_KEY")
    return ", ".join(active) if active else "(none)"
