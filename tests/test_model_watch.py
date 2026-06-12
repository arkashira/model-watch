import os
import json
import pytest
import sys
sys.path.insert(0, '../src')
from model_watch import deploy_model, detect_drift, log_instrumentation, ModelInput, ModelOutput

def test_deploy_model(tmp_path):
    output_dir = tmp_path / "deployment"
    deploy_model(str(output_dir))
    config_path = output_dir / "deployment_config.json"
    assert config_path.exists()
    with open(config_path, "r") as f:
        config = json.load(f)
    assert config["model_version"] == "1.0.0"
    assert config["status"] == "active"
    assert "/api/v1/predict" in config["endpoints"].values()

def test_detect_drift():
    baseline = 5.0
    assert not detect_drift([4.8, 5.1, 4.9], baseline)
    assert detect_drift([6.5, 7.0], baseline)

def test_log_instrumentation(tmp_path):
    input_data = ModelInput(
        features={"feature1": 0.5, "feature2": 0.7},
        metadata={"user_id": "123"}
    )
    output_data = ModelOutput(
        predictions=[0.8],
        confidence=0.95
    )
    log_path = tmp_path / "logs"
    log_instrumentation(input_data, output_data, str(log_path))
    log_file = log_path / "instrumentation.log"
    assert log_file.exists()
    with open(log_file, "r") as f:
        entry = json.loads(f.readline())
    assert entry["input"]["features"] == input_data.features
    assert entry["output"]["predictions"] == output_data.predictions
