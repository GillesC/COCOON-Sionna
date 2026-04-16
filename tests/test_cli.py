import json
from pathlib import Path

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
