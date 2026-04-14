from pathlib import Path

from cocoon_sionna.config import (
    AccessPointSpec,
    CoverageConfig,
    MobilityConfig,
    OptimizationConfig,
    OutputConfig,
    RadioConfig,
    ScenarioConfig,
    SceneConfig,
    SolverConfig,
)
from cocoon_sionna.pipeline import build_scene_only
from cocoon_sionna.scene_builder import SceneArtifacts


def test_build_scene_only_forces_osm_rebuild(monkeypatch, tmp_path: Path):
    scene_output_dir = tmp_path / "generated"
    scene_output_dir.mkdir()
    (scene_output_dir / "scene.xml").write_text("stale", encoding="utf-8")

    config = ScenarioConfig(
        name="test_osm",
        scene=SceneConfig(
            kind="osm",
            boundary_bbox=(3.706070, 51.058969, 3.712513, 51.060592),
            scene_output_dir=scene_output_dir,
            rebuild=False,
        ),
        access_point_spec=AccessPointSpec(),
        radio=RadioConfig(),
        coverage=CoverageConfig(),
        mobility=MobilityConfig(source="graph"),
        solver=SolverConfig(),
        optimization=OptimizationConfig(),
        candidate_sites_path=tmp_path / "sites.csv",
        outputs=OutputConfig(output_dir=tmp_path / "outputs"),
        scenario_path=tmp_path / "scenario.yaml",
    )

    called = {"value": False}

    class FakeBuilder:
        def __init__(self, scene_cfg):
            assert scene_cfg.rebuild is True

        def build(self):
            called["value"] = True
            return SceneArtifacts(
                scene_xml_path=scene_output_dir / "scene.xml",
                metadata_path=scene_output_dir / "scene_metadata.json",
                walk_graph_path=scene_output_dir / "walk_graph.json",
            )

    monkeypatch.setattr("cocoon_sionna.pipeline.OSMSceneBuilder", FakeBuilder)

    artifacts = build_scene_only(config)

    assert called["value"] is True
    assert artifacts.scene_xml_path == scene_output_dir / "scene.xml"
