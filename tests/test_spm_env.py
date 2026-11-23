"""`.env` parsing and, more importantly, precedence.

Precedence is the part worth pinning: `--set` > real environment > `.env` > JSON.
Getting it wrong is the kind of bug that silently runs a sweep against the wrong
model and is only noticed in the bill.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spm import env  # noqa: E402
from spm.config import Config  # noqa: E402

MANAGED = list(env.PROVIDER_KEYS) + ["SPM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"]


@pytest.fixture(autouse=True)
def clean_env():
    saved = {k: os.environ.pop(k, None) for k in MANAGED}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


# ----------------------------------------------------------------- parsing

def test_parse_handles_the_shapes_people_actually_write():
    parsed = env.parse(
        "# a comment\n"
        "\n"
        "SPM_ENGINE=cc/claude-opus-4-6\n"
        "  SPM_BASE_URL = http://localhost:20128/v1  \n"
        "export SPM_API_KEY=sk-abc123\n"
        'SPM_STRUCTURED="json"\n'
        "SPM_MOSAIC_PATH=''\n"
        "SPM_TEMPERATURE=0.7 # trailing comment\n"
        "MALFORMED_NO_EQUALS\n"
    )
    assert parsed["SPM_ENGINE"] == "cc/claude-opus-4-6"
    assert parsed["SPM_BASE_URL"] == "http://localhost:20128/v1"
    assert parsed["SPM_API_KEY"] == "sk-abc123"
    assert parsed["SPM_STRUCTURED"] == "json"
    assert parsed["SPM_MOSAIC_PATH"] == ""
    assert parsed["SPM_TEMPERATURE"] == "0.7"
    assert "MALFORMED_NO_EQUALS" not in parsed


def test_parse_keeps_hashes_inside_quoted_values_and_urls():
    """A '#' in a key or a URL fragment is data, not a comment."""
    parsed = env.parse('SPM_API_KEY="sk-a#b#c"\nSPM_BASE_URL=http://h/v1#x\n')
    assert parsed["SPM_API_KEY"] == "sk-a#b#c"
    assert parsed["SPM_BASE_URL"] == "http://h/v1#x"


# -------------------------------------------------------------- precedence

def test_a_real_environment_variable_beats_the_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("SPM_ENGINE=from-file\n")
    os.environ["SPM_ENGINE"] = "from-shell"
    env.load(str(tmp_path / ".env"))
    assert os.environ["SPM_ENGINE"] == "from-shell"


def test_dotenv_fills_in_what_the_shell_did_not_set(tmp_path):
    (tmp_path / ".env").write_text("SPM_ENGINE=from-file\nSPM_BASE_URL=http://x/v1\n")
    env.load(str(tmp_path / ".env"))
    assert os.environ["SPM_ENGINE"] == "from-file"
    assert os.environ["SPM_BASE_URL"] == "http://x/v1"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert env.load(str(tmp_path / "nope.env")) == {}


def test_env_overlays_the_json_config_and_types_numbers():
    raw = {"engine": "mock", "base_url": "", "temperature": 0.7, "n_agents": 30}
    os.environ.update(SPM_ENGINE="cc/claude-opus-4-6",
                      SPM_BASE_URL="http://localhost:20128/v1",
                      SPM_TEMPERATURE="0.3")
    cfg = Config.from_dict(env.apply(raw))
    assert cfg.engine == "cc/claude-opus-4-6"
    assert cfg.base_url == "http://localhost:20128/v1"
    assert cfg.temperature == 0.3 and isinstance(cfg.temperature, float)
    assert cfg.n_agents == 30, "experiment parameters must not be touched by .env"


def test_empty_env_value_does_not_blank_the_config():
    """`SPM_BASE_URL=` in a template should mean 'unset', not 'force empty'."""
    raw = {"base_url": "http://from-json/v1"}
    os.environ["SPM_BASE_URL"] = ""
    assert env.apply(raw)["base_url"] == "http://from-json/v1"


def test_command_line_set_beats_the_environment():
    import run_spm

    os.environ["SPM_ENGINE"] = "from-env"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        with open(path, "w") as f:
            json.dump({"engine": "mock"}, f)
        args = type("A", (), {"config": path, "set": ['engine="from-cli"']})()
        assert run_spm._load(args).engine == "from-cli"


def test_env_only_carries_provider_settings():
    """Nothing that changes a result may live in an untracked file."""
    experiment_fields = {"n_agents", "n_steps", "seed", "claim", "topology",
                         "attack", "defense"}
    assert not experiment_fields & set(env.PROVIDER_KEYS.values())


def test_example_template_parses_and_names_only_known_keys():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".env.example"), encoding="utf-8") as f:
        parsed = env.parse(f.read())
    assert parsed, "the template should contain live (uncommented) keys"
    known = set(env.PROVIDER_KEYS) | {"SPM_API_KEY"}
    assert set(parsed) <= known, f"template names unknown keys: {set(parsed) - known}"
    assert parsed["SPM_ENGINE"] == "mock", "template must default to the free backend"
