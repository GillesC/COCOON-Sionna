from cocoon_sionna.config import load_scenario_config


def test_load_etoile_config():
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    assert config.name == "etoile_demo"
    assert config.scene.kind == "builtin"
    assert config.access_point_spec.model == "wall_ap_default"
    assert config.radio.frequency_hz == 3.5e9
    assert config.radio.tx_power_dbm_ap == 28.0
    assert config.radio.ap_num_rows == 2
    assert config.radio.ap_num_cols == 3
    assert config.radio.ue_num_rows == 1
    assert config.radio.ue_num_cols == 1
    assert config.solver.enable_ray_tracing is True
    assert config.solver.require_gpu is False
    assert config.solver.synthetic_array is True
    assert config.optimization.enable_optimization is True
    assert config.candidate_sites_path.exists()
    assert config.optimization.max_candidate_ue_positions == 8
    assert config.optimization.baseline_site_ids == [
        "etoile_fixed_01",
        "etoile_fixed_02",
        "etoile_fixed_03",
        "etoile_fixed_04",
    ]
    assert config.mobility.speed_variation_fraction == 0.12
    assert config.optimization.relocation_interval_s == 15.0


def test_load_rabot_boundary_bbox():
    config = load_scenario_config("scenarios/rabot.yaml")
    assert config.scene.kind == "osm"
    assert config.scene.boundary_bbox is not None
    west, south, east, north = config.scene.boundary_bbox
    assert west < east
    assert south < north
    assert config.scene.boundary_path is None
    assert config.outputs.scene_animation_speedup == 10.0
    assert config.solver.require_gpu is True
