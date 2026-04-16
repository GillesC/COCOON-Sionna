from pathlib import Path

from cocoon_sionna.config import (
    AccessPointSpec,
    CoverageConfig,
    MobilityConfig,
    OutputConfig,
    PlacementConfig,
    RadioConfig,
    ScenarioConfig,
    SceneConfig,
    SolverConfig,
)
from cocoon_sionna.pipeline import build_scene_only, run_scenario
from cocoon_sionna.scene_builder import OSMSceneBuilder, SceneArtifacts


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
        placement=PlacementConfig(),
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


def test_osm_scene_builder_preserves_existing_meshes_when_fetch_fails(monkeypatch, tmp_path: Path):
    scene_output_dir = tmp_path / "generated"
    mesh_dir = scene_output_dir / "meshes"
    mesh_dir.mkdir(parents=True)
    old_mesh = mesh_dir / "ground.ply"
    old_mesh.write_text("old", encoding="utf-8")

    scene_cfg = SceneConfig(
        kind="osm",
        boundary_bbox=(3.706070, 51.058969, 3.712513, 51.060592),
        scene_output_dir=scene_output_dir,
        rebuild=True,
    )

    def _fail_fetch(self, _boundary_lonlat):
        raise RuntimeError("overpass unavailable")

    monkeypatch.setattr("cocoon_sionna.scene_builder.OverpassClient.fetch", _fail_fetch)

    try:
        OSMSceneBuilder(scene_cfg).build()
    except RuntimeError as exc:
        assert "overpass unavailable" in str(exc)
    else:
        raise AssertionError("Expected the scene build to fail")

    assert old_mesh.exists()
    assert old_mesh.read_text(encoding="utf-8") == "old"


def test_run_scenario_requires_prebuilt_osm_scene_assets(tmp_path: Path):
    config = ScenarioConfig(
        name="test_osm_missing",
        scene=SceneConfig(
            kind="osm",
            boundary_bbox=(3.706070, 51.058969, 3.712513, 51.060592),
            scene_output_dir=tmp_path / "generated",
            rebuild=False,
        ),
        access_point_spec=AccessPointSpec(),
        radio=RadioConfig(),
        coverage=CoverageConfig(enabled=False),
        mobility=MobilityConfig(source="graph"),
        solver=SolverConfig(enable_ray_tracing=False),
        placement=PlacementConfig(),
        candidate_sites_path=tmp_path / "sites.csv",
        outputs=OutputConfig(output_dir=tmp_path / "outputs"),
        scenario_path=tmp_path / "scenario.yaml",
    )

    try:
        run_scenario(config)
    except FileNotFoundError as exc:
        assert "build-scene" in str(exc)
    else:
        raise AssertionError("Expected run_scenario() to require prebuilt OSM assets")
