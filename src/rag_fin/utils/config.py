from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_STAGES = {"parser", "retrieval", "generation", "eval"}


def project_root() -> Path:
    """Return repository root from the src package path."""
    return Path(__file__).resolve().parents[3]


def resolve_config_path(
    stage: str,
    name: str = "default",
    config_root: Path | None = None,
) -> Path:
    """Resolve config path like configs/<stage>/<name>.json."""
    normalized_stage = stage.strip().lower()
    if normalized_stage not in VALID_STAGES:
        allowed = ", ".join(sorted(VALID_STAGES))
        raise ValueError(f"unsupported stage '{stage}'. Allowed: {allowed}")

    root = config_root if config_root is not None else project_root() / "configs"
    return root / normalized_stage / f"{name}.json"


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file into a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"config at {config_path} must be a JSON object")
    return payload


def load_stage_config(
    stage: str,
    name: str = "default",
    config_root: Path | None = None,
) -> dict[str, Any]:
    """Load stage config by stage + config name."""
    path = resolve_config_path(stage=stage, name=name, config_root=config_root)
    return load_json_config(path)
