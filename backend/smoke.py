"""End-to-end smoke test for the backend.

Spins up FastAPI in-process (no uvicorn needed), hits each major endpoint with
realistic inputs, and asserts shape + non-emptiness. Single ML inference is
required (one /predict/draft hit) so we exercise the model registry too.

    uv run python -m backend.smoke

Exit code 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys
import time

from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def step(name: str, fn) -> None:
    t = time.time()
    try:
        fn()
    except AssertionError as e:
        print(f"  ✗ {name}  ({e})")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ {name}  ({type(e).__name__}: {e})")
        sys.exit(1)
    print(f"  ✓ {name}  ({time.time() - t:.1f}s)")


# ----- step bodies -----------------------------------------------------------


def s_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200, r.status_code
    assert r.json()["status"] == "ok"


def s_checkpoints() -> dict:
    r = client.get("/api/checkpoints")
    assert r.status_code == 200, r.status_code
    j = r.json()
    assert len(j) >= 1, "no checkpoints found"
    state["ckpts"] = j


def s_card_detail() -> None:
    r = client.get("/api/cards/0")
    assert r.status_code == 200, r.status_code


def s_card_search() -> None:
    r = client.get("/api/cards/search", params={"q": "Lightning Bolt", "limit": 5})
    assert r.status_code == 200, r.status_code
    j = r.json()
    assert any(it["name"] == "Lightning Bolt" for it in j["items"]), j


def s_pick_random() -> None:
    r = client.get("/api/picks/random/one")
    assert r.status_code == 200, r.status_code
    j = r.json()
    assert j["pack"], "empty pack"
    state["pick"] = j


def s_cube_groups() -> None:
    r = client.get("/api/picks/cubes/list", params={"limit": 5})
    assert r.status_code == 200, r.status_code


def s_metrics_run() -> None:
    r = client.get("/api/metrics/runs")
    assert r.status_code == 200, r.status_code
    runs = r.json()["runs"]
    assert runs, "no runs found"
    rr = client.get(f"/api/metrics/runs/{runs[0]}")
    assert rr.status_code == 200, rr.status_code


def s_predict_draft() -> None:
    # Pick a checkpoint that's a real run (not prod, faster).
    ckpt = next((c for c in state["ckpts"] if c["kind"] in ("ckpt", "best", "latest")), state["ckpts"][0])
    body = {
        "ckpt": ckpt["key"],
        "pool": state["pick"]["pool"],
        "pack": state["pick"]["pack"],
    }
    r = client.post("/api/predict/draft", json=body)
    assert r.status_code == 200, (r.status_code, r.text)
    j = r.json()
    assert len(j["ranked"]) == len(set(body["pack"])), (
        f"ranked count {len(j['ranked'])} != pack {len(set(body['pack']))}"
    )


def s_predict_deckbuilder() -> None:
    # Use a val deck pool (mainboard ∪ sideboard).
    r = client.get("/api/decks/0")
    assert r.status_code == 200, r.status_code
    deck = r.json()
    pool = list((deck["mainboard"] or []) + (deck["sideboard"] or []))
    ckpt = next((c for c in state["ckpts"] if c["kind"] in ("ckpt", "best", "latest")), state["ckpts"][0])
    body = {
        "ckpt": ckpt["key"],
        "pool": pool,
        "max_spells": 23,
        "max_lands": 17,
        "seed_count": 10,
    }
    r = client.post("/api/predict/deckbuilder", json=body)
    assert r.status_code == 200, (r.status_code, r.text)
    j = r.json()
    assert 0 < len(j["deck"]) <= 40, len(j["deck"])


state: dict = {}


def main() -> int:
    print("backend smoke test")
    step("health", s_health)
    step("checkpoints list", s_checkpoints)
    step("card detail", s_card_detail)
    step("card search", s_card_search)
    step("pick random", s_pick_random)
    step("cube groups", s_cube_groups)
    step("metrics run", s_metrics_run)
    step("predict draft", s_predict_draft)
    step("predict deckbuilder", s_predict_deckbuilder)
    print("\nall green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
