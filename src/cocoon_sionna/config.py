"""Scenario configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass(slots=True)
class SceneConfig:
    kind: str
    sionna_scene: str | None = None
    boundary_path: Path | None = None
    boundary_bbox: tuple[float, float, float, float] | None = None
    scene_output_dir: Path | None = None
    scene_xml_path: Path | None = None
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    default_building_height_m: float = 12.0
    building_height_per_level_m: float = 3.0
    material_ground: str = "concrete"
    material_wall: str = "concrete"
    material_roof: str = "metal"
    rebuild: bool = False


@dataclass(slots=True)
class AccessPointSpec:
    model: str = "default_ap"
    tx_power_dbm: float = 30.0
    num_rows: int = 2
    num_cols: int = 3
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    array_pattern: str = "tr38901"
    array_polarization: str = "VH"
    synthetic_array: bool | None = None


@dataclass(slots=True)
class RadioConfig:
    frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    cfr_bins: int = 256
    sampling_frequency_hz: float | None = None
    ap_num_rows: int = 2
    ap_num_cols: int = 3
    ue_num_rows: int = 1
    ue_num_cols: int = 1
    array_pattern: str = "tr38901"
    array_polarization: str = "VH"
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    tx_power_dbm_ap: float = 30.0
    tx_power_dbm_ue: float = 18.0
    ue_height_m: float = 1.5
    temperature_k: float = 290.0

    def frequencies(self) -> list[float]:
        half_bw = 0.5 * self.bandwidth_hz
        step = self.bandwidth_hz / self.cfr_bins
        return [(-half_bw + i * step) for i in range(self.cfr_bins)]

    @property
    def effective_sampling_frequency_hz(self) -> float:
        return self.sampling_frequency_hz or self.bandwidth_hz


@dataclass(slots=True)
class CoverageConfig:
    enabled: bool = True
    cell_size_m: tuple[float, float] = (10.0, 10.0)
    height_m: float = 1.5
    center_m: tuple[float, float, float] | None = None
    size_m: tuple[float, float] | None = None


@dataclass(slots=True)
class MobilityConfig:
    source: str
    graph_path: Path | None = None
    num_users: int = 6
    duration_s: float = 30.0
    step_s: float = 1.0
    allow_open_area: bool = True
    open_area_grid_spacing_m: float = 8.0
    open_area_connection_radius_m: float = 14.0
    open_area_clearance_m: float = 0.75
    speed_mps_range: tuple[float, float] = (0.9, 1.6)
    pedestrian_speed_mps_range: tuple[float, float] | None = None
    bike_speed_mps_range: tuple[float, float] | None = None
    bike_fraction: float = 0.0
    speed_variation_fraction: float = 0.15
    dwell_s_range: tuple[float, float] = (4.0, 12.0)
    entry_node_buffer_m: float = 5.0
    seed: int = 7


@dataclass(slots=True)
class SolverConfig:
    enable_ray_tracing: bool = True
    require_gpu: bool = False
    path_max_depth: int = 3
    radio_map_max_depth: int = 3
    samples_per_src: int = 10000
    samples_per_tx: int = 10000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    seed: int = 42


@dataclass(slots=True)
class PlacementConfig:
    num_fixed_aps: int = 0
    num_movable_aps: int = 3
    sinr_threshold_db: float = 3.0
    outage_weight: float = 1.0
    percentile_weight: float = 0.05
    peer_tiebreak_weight: float = 0.01
    candidate_min_spacing_m: float = 8.0
    candidate_wall_spacing_m: float = 18.0
    candidate_corner_clearance_m: float = 2.0
    candidate_wall_height_m: float = 1.5
    candidate_wall_offset_m: float = 0.25
    window_interval_s: float = 6.0
    historical_csi_decay_rate_per_s: float = math.log(2.0) / 6.0
    random_seed: int = 7
    heuristic_k_nearest: int = 8
    optimization_2_distance_threshold_m: float = 25.0
    exact_max_iterations: int = 50000


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path
    scene_animation_speedup: float = 1.0
    write_csi_exports: bool = True
    enable_csi_cache: bool = True


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    scene: SceneConfig
    access_point_spec: AccessPointSpec
    radio: RadioConfig
    coverage: CoverageConfig
    mobility: MobilityConfig
    solver: SolverConfig
    placement: PlacementConfig
    candidate_sites_path: Path
    outputs: OutputConfig
    scenario_path: Path = field(repr=False)


def _tuple_of_floats(value: Any, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if len(value) != size:
        raise ValueError(f"Expected {size} values, got {value!r}")
    return tuple(float(v) for v in value)


def _load_scene(base_dir: Path, data: dict[str, Any]) -> SceneConfig:
    return SceneConfig(
        kind=str(data["kind"]),
        sionna_scene=data.get("sionna_scene"),
        boundary_path=_resolve_path(base_dir, data.get("boundary_path")),
        boundary_bbox=(
            _tuple_of_floats(data["boundary_bbox"], 4, (0.0, 0.0, 0.0, 0.0))
            if data.get("boundary_bbox") is not None
            else None
        ),
        scene_output_dir=_resolve_path(base_dir, data.get("scene_output_dir")),
        scene_xml_path=_resolve_path(base_dir, data.get("scene_xml_path")),
        overpass_url=str(data.get("overpass_url", "https://overpass-api.de/api/interpreter")),
        default_building_height_m=float(data.get("default_building_height_m", 12.0)),
        building_height_per_level_m=float(data.get("building_height_per_level_m", 3.0)),
        material_ground=str(data.get("material_ground", "concrete")),
        material_wall=str(data.get("material_wall", "concrete")),
        material_roof=str(data.get("material_roof", "metal")),
        rebuild=bool(data.get("rebuild", False)),
    )


def _load_access_point_spec(base_dir: Path, raw: dict[str, Any]) -> AccessPointSpec:
    path = _resolve_path(base_dir, raw.get("access_point_spec_path"))
    if path is None:
        default_path = (base_dir / "access_point_spec.yml").resolve()
        path = default_path if default_path.exists() else None

    spec_data: dict[str, Any] = {}
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping in access point spec file {path}")
        spec_data = loaded

    return AccessPointSpec(
        model=str(spec_data.get("model", "default_ap")),
        tx_power_dbm=float(spec_data.get("tx_power_dbm", 30.0)),
        num_rows=int(spec_data.get("num_rows", 2)),
        num_cols=int(spec_data.get("num_cols", 3)),
        vertical_spacing=float(spec_data.get("vertical_spacing", 0.5)),
        horizontal_spacing=float(spec_data.get("horizontal_spacing", 0.5)),
        array_pattern=str(spec_data.get("array_pattern", "tr38901")),
        array_polarization=str(spec_data.get("array_polarization", "VH")),
        synthetic_array=(
            bool(spec_data["synthetic_array"])
            if spec_data.get("synthetic_array") is not None
            else None
        ),
    )


def _load_radio(data: dict[str, Any], access_point_spec: AccessPointSpec) -> RadioConfig:
    return RadioConfig(
        frequency_hz=float(data.get("frequency_hz", 3.5e9)),
        bandwidth_hz=float(data.get("bandwidth_hz", 100e6)),
        cfr_bins=int(data.get("cfr_bins", 256)),
        sampling_frequency_hz=(
            float(data["sampling_frequency_hz"])
            if data.get("sampling_frequency_hz") is not None
            else None
        ),
        ap_num_rows=int(data.get("ap_num_rows", access_point_spec.num_rows)),
        ap_num_cols=int(data.get("ap_num_cols", access_point_spec.num_cols)),
        ue_num_rows=int(data.get("ue_num_rows", 1)),
        ue_num_cols=int(data.get("ue_num_cols", 1)),
        array_pattern=str(data.get("array_pattern", access_point_spec.array_pattern)),
        array_polarization=str(data.get("array_polarization", access_point_spec.array_polarization)),
        vertical_spacing=float(data.get("vertical_spacing", access_point_spec.vertical_spacing)),
        horizontal_spacing=float(data.get("horizontal_spacing", access_point_spec.horizontal_spacing)),
        tx_power_dbm_ap=float(data.get("tx_power_dbm_ap", access_point_spec.tx_power_dbm)),
        tx_power_dbm_ue=float(data.get("tx_power_dbm_ue", 18.0)),
        ue_height_m=float(data.get("ue_height_m", 1.5)),
        temperature_k=float(data.get("temperature_k", 290.0)),
    )


def _load_coverage(data: dict[str, Any]) -> CoverageConfig:
    return CoverageConfig(
        enabled=bool(data.get("enabled", True)),
        cell_size_m=_tuple_of_floats(data.get("cell_size_m"), 2, (10.0, 10.0)),
        height_m=float(data.get("height_m", 1.5)),
        center_m=(
            _tuple_of_floats(data["center_m"], 3, (0.0, 0.0, 0.0))
            if data.get("center_m") is not None
            else None
        ),
        size_m=(
            _tuple_of_floats(data["size_m"], 2, (1.0, 1.0))
            if data.get("size_m") is not None
            else None
        ),
    )


def _load_mobility(base_dir: Path, data: dict[str, Any]) -> MobilityConfig:
    return MobilityConfig(
        source=str(data["source"]),
        graph_path=_resolve_path(base_dir, data.get("graph_path")),
        num_users=int(data.get("num_users", 6)),
        duration_s=float(data.get("duration_s", 30.0)),
        step_s=float(data.get("step_s", 1.0)),
        allow_open_area=bool(data.get("allow_open_area", True)),
        open_area_grid_spacing_m=float(data.get("open_area_grid_spacing_m", 8.0)),
        open_area_connection_radius_m=float(data.get("open_area_connection_radius_m", 14.0)),
        open_area_clearance_m=float(data.get("open_area_clearance_m", 0.75)),
        speed_mps_range=_tuple_of_floats(data.get("speed_mps_range"), 2, (0.9, 1.6)),
        pedestrian_speed_mps_range=(
            _tuple_of_floats(data["pedestrian_speed_mps_range"], 2, (0.9, 1.6))
            if data.get("pedestrian_speed_mps_range") is not None
            else None
        ),
        bike_speed_mps_range=(
            _tuple_of_floats(data["bike_speed_mps_range"], 2, (3.0, 5.5))
            if data.get("bike_speed_mps_range") is not None
            else None
        ),
        bike_fraction=float(data.get("bike_fraction", 0.0)),
        speed_variation_fraction=float(data.get("speed_variation_fraction", 0.15)),
        dwell_s_range=_tuple_of_floats(data.get("dwell_s_range"), 2, (4.0, 12.0)),
        entry_node_buffer_m=float(data.get("entry_node_buffer_m", 5.0)),
        seed=int(data.get("seed", 7)),
    )


def _load_solver(data: dict[str, Any], access_point_spec: AccessPointSpec) -> SolverConfig:
    return SolverConfig(
        enable_ray_tracing=bool(data.get("enable_ray_tracing", True)),
        require_gpu=bool(data.get("require_gpu", False)),
        path_max_depth=int(data.get("path_max_depth", 3)),
        radio_map_max_depth=int(data.get("radio_map_max_depth", 3)),
        samples_per_src=int(data.get("samples_per_src", 10000)),
        samples_per_tx=int(data.get("samples_per_tx", 10000)),
        synthetic_array=bool(
            data.get(
                "synthetic_array",
                True if access_point_spec.synthetic_array is None else access_point_spec.synthetic_array,
            )
        ),
        los=bool(data.get("los", True)),
        specular_reflection=bool(data.get("specular_reflection", True)),
        diffuse_reflection=bool(data.get("diffuse_reflection", False)),
        refraction=bool(data.get("refraction", True)),
        diffraction=bool(data.get("diffraction", False)),
        edge_diffraction=bool(data.get("edge_diffraction", False)),
        diffraction_lit_region=bool(data.get("diffraction_lit_region", True)),
        seed=int(data.get("seed", 42)),
    )


def _load_placement(data: dict[str, Any]) -> PlacementConfig:
    window_interval_s = float(data.get("window_interval_s", 6.0))
    return PlacementConfig(
        num_fixed_aps=int(data.get("num_fixed_aps", 0)),
        num_movable_aps=int(data.get("num_movable_aps", 3)),
        sinr_threshold_db=float(data.get("sinr_threshold_db", 3.0)),
        outage_weight=float(data.get("outage_weight", 1.0)),
        percentile_weight=float(data.get("percentile_weight", 0.05)),
        peer_tiebreak_weight=float(data.get("peer_tiebreak_weight", 0.01)),
        candidate_min_spacing_m=float(data.get("candidate_min_spacing_m", 8.0)),
        candidate_wall_spacing_m=float(data.get("candidate_wall_spacing_m", 18.0)),
        candidate_corner_clearance_m=float(data.get("candidate_corner_clearance_m", 2.0)),
        candidate_wall_height_m=float(data.get("candidate_wall_height_m", 1.5)),
        candidate_wall_offset_m=float(data.get("candidate_wall_offset_m", 0.25)),
        window_interval_s=window_interval_s,
        historical_csi_decay_rate_per_s=float(
            data.get("historical_csi_decay_rate_per_s", math.log(2.0) / max(window_interval_s, 1e-9))
        ),
        random_seed=int(data.get("random_seed", 7)),
        heuristic_k_nearest=int(data.get("heuristic_k_nearest", 8)),
        optimization_2_distance_threshold_m=float(data.get("optimization_2_distance_threshold_m", 25.0)),
        exact_max_iterations=int(data.get("exact_max_iterations", 50000)),
    )


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    scenario_path = Path(path).resolve()
    base_dir = scenario_path.parent.parent if scenario_path.parent.name == "scenarios" else scenario_path.parent
    with scenario_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    access_point_spec = _load_access_point_spec(base_dir, raw)
    outputs_dir = _resolve_path(base_dir, raw["outputs"]["output_dir"])
    assert outputs_dir is not None
    return ScenarioConfig(
        name=str(raw["name"]),
        scene=_load_scene(base_dir, raw["scene"]),
        access_point_spec=access_point_spec,
        radio=_load_radio(raw.get("radio", {}), access_point_spec),
        coverage=_load_coverage(raw.get("coverage", {})),
        mobility=_load_mobility(base_dir, raw["mobility"]),
        solver=_load_solver(raw.get("solver", {}), access_point_spec),
        placement=_load_placement(raw.get("placement", {})),
        candidate_sites_path=_resolve_path(base_dir, raw["candidate_sites_path"]),
        outputs=OutputConfig(
            output_dir=outputs_dir,
            scene_animation_speedup=float(raw["outputs"].get("scene_animation_speedup", 1.0)),
            write_csi_exports=bool(raw["outputs"].get("write_csi_exports", True)),
            enable_csi_cache=bool(raw["outputs"].get("enable_csi_cache", True)),
        ),
        scenario_path=scenario_path,
    )
