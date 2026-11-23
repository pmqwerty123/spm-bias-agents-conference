"""Gateway integration tests, run against a stand-in for a 9Router-style endpoint.

The stub deliberately behaves the way a routing gateway does and OpenAI does not:

* it rejects ``response_format`` with a 400, the way an endpoint that never
  implemented strict structured output would;
* it wraps the JSON object in prose and markdown fences, the way a model behind a
  router that was not asked for strict JSON does;
* it returns a plain refusal on the first call to one prompt, so the repair retry
  is exercised.

No network and no API key. Run: python -m pytest tests/test_spm_gateway.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
logging.disable(logging.INFO)

from spm import runner                              # noqa: E402
from spm.config import Config                       # noqa: E402
from spm.llm import ChatBackend, TurnContext, extract_json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE = {"calls": 0, "saw_response_format": 0, "auth": None, "models": set()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["calls"] += 1
        STATE["auth"] = self.headers.get("Authorization")
        STATE["models"].add(body.get("model"))

        if "response_format" in body:
            STATE["saw_response_format"] += 1
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"response_format is not supported"}}')
            return

        if STATE["calls"] == 2:                     # force one repair round trip
            content = "I'd rather not put a number on it."
        else:
            content = (
                "Sure, here's my read:\n\n```json\n"
                '{"confidence": 0.63, "stance": "My meter ran above the rated '
                'figure again, so I am leaning yes."}\n```\nHope that helps!'
            )

        payload = {
            "id": "chatcmpl-stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 640, "completion_tokens": 41, "total_tokens": 681},
            "model": body.get("model", "stub"),
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture(scope="module")
def gateway():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset():
    STATE.update(calls=0, saw_response_format=0, auth=None, models=set())


def base_config(**over) -> Config:
    with open(os.path.join(REPO, "configs", "spm_experiment.json")) as f:
        raw = json.load(f)
    raw.update(over)
    return Config.from_dict(raw)


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("reply", [
    '{"confidence": 0.7, "stance": "yes"}',
    '```json\n{"confidence": 0.7, "stance": "yes"}\n```',
    'Sure! {"confidence": 0.7, "stance": "yes"} Let me know if you want more.',
    'Thinking...\n{"confidence": 0.7, "stance": "a } brace inside a string"}\ndone',
])
def test_extract_json_survives_gateway_prose(reply):
    assert extract_json(reply)["confidence"] == 0.7


def test_extract_json_raises_on_prose_only():
    with pytest.raises(ValueError):
        extract_json("I'd rather not say.")


# ------------------------------------------------------------------ backend

def test_auto_mode_picks_json_when_base_url_is_set(gateway):
    assert ChatBackend("m", 0.7, api_key="k", base_url=gateway).structured == "json"
    assert ChatBackend("m", 0.7, api_key="k").structured == "native"


def test_gateway_needs_no_api_key(gateway):
    """A self-hosted router with REQUIRE_API_KEY unset still has to work."""
    for var in ("SPM_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(var, None)
    backend = ChatBackend("cc/claude-opus-4-6", 0.7, base_url=gateway)
    resp = backend.elicit("sys", "prompt", TurnContext(0.5, 0.2, []))
    assert 0.0 < resp.confidence < 1.0
    assert STATE["auth"] == "Bearer not-required"


def test_json_mode_never_sends_response_format(gateway):
    backend = ChatBackend("auto", 0.7, api_key="k", base_url=gateway)
    backend.elicit("sys", "prompt", TurnContext(0.5, 0.2, []))
    assert STATE["saw_response_format"] == 0
    assert STATE["models"] == {"auto"}


def test_json_mode_parses_fenced_reply_and_records_usage(gateway):
    backend = ChatBackend("m", 0.7, api_key="k", base_url=gateway)
    resp = backend.elicit("sys", "prompt", TurnContext(0.5, 0.2, []))
    assert resp.confidence == pytest.approx(0.63)
    assert "meter" in resp.stance
    assert backend.prompt_tokens == 640 and backend.completion_tokens == 41


def test_repair_retry_recovers_from_a_refusal(gateway):
    backend = ChatBackend("m", 0.7, api_key="k", base_url=gateway)
    backend.elicit("sys", "first", TurnContext(0.5, 0.2, []))    # call 1: fine
    resp = backend.elicit("sys", "second", TurnContext(0.5, 0.2, []))  # call 2 refuses
    assert 0.0 < resp.confidence < 1.0
    assert STATE["calls"] == 3, "the refusal should have been repaired, not retried blind"


def test_native_mode_fails_loudly_with_actionable_advice(gateway):
    backend = ChatBackend("m", 0.7, api_key="k", base_url=gateway, structured="native")
    with pytest.raises(RuntimeError, match=r'(?s)structured.*json'):
        backend.elicit("sys", "prompt", TurnContext(0.5, 0.2, []))
    assert STATE["saw_response_format"] > 0


# ------------------------------------------------------------------ full run

def test_full_simulation_through_the_gateway(gateway):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = base_config(run_id="gw", n_agents=6, n_steps=4, out_dir=tmp,
                          engine="cc/claude-opus-4-6", base_url=gateway)
        cfg.attack.fraction = 0.34
        summary = runner.run(cfg)
        assert summary["n_manipulators"] == 2
        # Manipulators are programmatic, so only honest agents cost calls -- plus
        # the one repair round trip the stub forces, which is why this is a floor
        # rather than an equality. Budget from run.json, not from N * T.
        turns = summary["n_honest"] * 4
        assert turns <= summary["model_calls"] <= turns + 2
        assert summary["prompt_tokens"] > 0
        assert os.path.exists(os.path.join(tmp, "gw", "simulation.db"))
