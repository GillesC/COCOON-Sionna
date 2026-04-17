import math

from cocoon_sionna.config import load_scenario_config


def test_load_rabot_config():
    config = load_scenario_config("scenarios/rabot.yaml")
    assert config.name == "rabot_outdoor"
    assert config.scene.kind == "osm"
    assert config.access_point_spec.model == "wall_ap_default"
    assert config.radio.frequency_hz == 3.5e9
    assert config.radio.tx_power_dbm_ap == 28.0
    assert config.radio.ap_num_rows == 2
    assert config.radio.ap_num_cols == 3
    assert config.radio.ue_num_rows == 1
    assert config.radio.ue_num_cols == 1
    assert config.coverage.enabled is False
    assert config.solver.enable_ray_tracing is True
    assert config.solver.require_gpu is True
    assert config.solver.synthetic_array is True
    assert config.placement.num_fixed_aps == 0
    assert config.placement.num_movable_aps == 6
    assert config.placement.enable_optimization_1 is True
    assert config.placement.enable_optimization_2 is True
    assert config.placement.enable_optimization_3 is True
    assert config.candidate_sites_path.exists()
    assert config.placement.heuristic_k_nearest == 10
    assert config.placement.optimization_2_distance_threshold_m == 25.0
    assert config.mobility.speed_variation_fraction == 0.15
    assert config.placement.window_interval_s == 60.0
    assert math.isclose(config.placement.historical_csi_decay_rate_per_s, math.log(2.0) / 600.0)
    assert config.outputs.write_csi_exports is False
    assert config.outputs.enable_csi_cache is False


def test_load_rabot_boundary_bbox():
    config = load_scenario_config("scenarios/rabot.yaml")
    assert config.scene.kind == "osm"
    assert config.scene.boundary_bbox is not None
    west, south, east, north = config.scene.boundary_bbox
    assert west < east
    assert south < north
    assert config.scene.boundary_path is None
    assert config.outputs.scene_animation_speedup == 200.0
    assert config.coverage.enabled is False
    assert config.outputs.write_csi_exports is False
    assert config.outputs.enable_csi_cache is False
    assert config.placement.num_fixed_aps == 0
    assert config.placement.num_movable_aps == 6
    assert config.solver.require_gpu is True
