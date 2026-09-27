"""Guards against the deployed app drifting from the environment the model was trained in."""
import json

import yaml

from src.config import ROOT


def read_pins(path) -> dict:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.lower()] = version
    return pins


def test_app_requirements_match_model_training_versions():
    app_pins = read_pins(ROOT / "app" / "requirements.txt")
    trained_with = json.loads((ROOT / "models" / "metadata.json").read_text())["library_versions"]
    assert app_pins["scikit-learn"] == trained_with["scikit-learn"]
    if "xgboost" in trained_with:
        assert app_pins["xgboost-cpu"] == trained_with["xgboost"]
    if "lightgbm" in trained_with:
        assert app_pins["lightgbm"] == trained_with["lightgbm"]


def test_app_requirements_match_training_requirements():
    app_pins = read_pins(ROOT / "app" / "requirements.txt")
    train_pins = read_pins(ROOT / "requirements.txt")
    for package in ("scikit-learn", "numpy", "pandas", "joblib", "pyyaml", "gradio"):
        assert app_pins[package] == train_pins[package], package


def test_render_blueprint_points_at_existing_files():
    service = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    assert service["plan"] == "free"
    assert (ROOT / service["buildCommand"].split()[-1]).exists()
    assert (ROOT / service["startCommand"].split()[-1]).exists()
    env = {item["key"]: item["value"] for item in service["envVars"]}
    assert env["PYTHON_VERSION"].startswith("3.10")
