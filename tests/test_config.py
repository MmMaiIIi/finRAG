"""Tests for YAML config loading."""

from pathlib import Path

from src.utils.config import load_yaml_config


def test_load_model_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs" / "model.yaml")

    assert "embedding" in config
    assert "generation" in config
    assert config["embedding"]["provider"] == "local_stub"


def test_load_retrieval_and_eval_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    retrieval = load_yaml_config(root / "configs" / "retrieval.yaml")
    eval_cfg = load_yaml_config(root / "configs" / "eval.yaml")

    assert retrieval["retrieval"]["top_k"] > 0
    assert "metrics" in eval_cfg
    assert isinstance(eval_cfg["metrics"], list)

