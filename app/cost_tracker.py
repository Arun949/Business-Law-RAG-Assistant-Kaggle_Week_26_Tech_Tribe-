import json
import os
from datetime import datetime

COST_FILE = "data/cost_tracker.json"

COSTS = {
    "input":  0.150,   # $/1M tokens  — gpt-4o-mini prompt tokens
    "output": 0.600,   # $/1M tokens  — gpt-4o-mini completion tokens
    "embed":  0.020,   # $/1M tokens  — text-embedding-3-small
}


def _load() -> dict:
    if os.path.exists(COST_FILE):
        with open(COST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0, "by_script": {}, "last_updated": None}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(COST_FILE), exist_ok=True)
    with open(COST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def track_cost(response, is_embedding: bool = False, script: str = "") -> float:
    usage = response.usage

    if is_embedding:
        cost = usage.total_tokens * COSTS["embed"] / 1_000_000
    else:
        cost = (
            usage.prompt_tokens     * COSTS["input"] +
            usage.completion_tokens * COSTS["output"]
        ) / 1_000_000

    cost = round(cost, 6)

    # Auto-detect calling script if not provided
    if not script:
        import inspect
        frame = inspect.stack()[1]
        script = os.path.splitext(os.path.basename(frame.filename))[0]

    data = _load()
    data["total_usd"]   = round(data.get("total_usd", 0.0) + cost, 6)
    data["calls"]       = data.get("calls", 0) + 1
    data["last_updated"] = datetime.now().isoformat(timespec="seconds")

    by_script = data.setdefault("by_script", {})
    by_script[script] = round(by_script.get(script, 0.0) + cost, 6)

    _save(data)
    return cost


def get_total() -> float:
    """Return the cumulative total cost (USD) from cost_tracker.json."""
    return _load().get("total_usd", 0.0)


def reset() -> None:
    """Reset the cost tracker to zero."""
    _save({"total_usd": 0.0, "calls": 0, "by_script": {},
           "last_updated": datetime.now().isoformat(timespec="seconds")})
    print(f"Cost tracker reset. ({COST_FILE})")
