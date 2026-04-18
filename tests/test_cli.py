import json
from pathlib import Path
from types import SimpleNamespace

from cocoon_sionna import cli


def test_cli_postprocess_dispatches_bundle(monkeypatch, capsys, tmp_path: Path):
    observed = {}

    def _fake_bundle(**kwargs):
        observed["kwargs"] = kwargs
        return {"manifest": tmp_path / "out" / "postprocessing" / "visualization_manifest.json"}

    monkeypatch.setattr("cocoon_sionna.cli.resolve_output_dir_argument", lambda target: tmp_path / "out")
    monkeypatch.setattr("cocoon_sionna.cli.configure_logging", lambda level, path: observed.update({"log_level": level, "log_path": path}))
    monkeypatch.setattr("cocoon_sionna.cli.run_visualization_postprocess", _fake_bundle)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cocoon-sionna",
            "postprocess",
            "outputs/rabot",
            "--threshold-max-db",
            "15",
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert observed["log_path"] == tmp_path / "out" / "postprocess.log"
    assert observed["kwargs"]["output_dir"] == tmp_path / "out"
    assert observed["kwargs"]["threshold_max_db"] == 15.0
    assert "manifest" in payload


def test_cli_all_dispatches_run_then_postprocess(monkeypatch, capsys, tmp_path: Path):
    observed = {}
    fake_config = SimpleNamespace(
        outputs=SimpleNamespace(output_dir=tmp_path / "out"),
        scene=SimpleNamespace(scene_output_dir=None),
    )

    def _fake_load_config(scenario):
        observed["scenario"] = scenario
        return fake_config

    def _fake_run(config):
        observed["run_config"] = config
        return {"summary": "ok"}

    monkeypatch.setattr("cocoon_sionna.cli.configure_logging", lambda level, path: observed.update({"log_level": level, "log_path": path}))
    monkeypatch.setattr("cocoon_sionna.cli.load_scenario_config", _fake_load_config)
    monkeypatch.setattr("cocoon_sionna.cli.run_scenario", _fake_run)
    monkeypatch.setattr(
        "cocoon_sionna.cli.run_visualization_postprocess",
        lambda **kwargs: observed.update({"postprocess_kwargs": kwargs}) or {"manifest": tmp_path / "out" / "postprocessing" / "visualization_manifest.json"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cocoon-sionna",
            "all",
            "scenarios/rabot.yaml",
            "--threshold-step-db",
            "2",
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert observed["scenario"] == "scenarios/rabot.yaml"
    assert observed["run_config"] is fake_config
    assert observed["log_path"] == tmp_path / "out" / "all.log"
    assert observed["postprocess_kwargs"]["output_dir"] == tmp_path / "out"
    assert observed["postprocess_kwargs"]["threshold_step_db"] == 2.0
    assert payload["run"]["summary"] == "ok"
    assert "manifest" in payload["postprocess"]
