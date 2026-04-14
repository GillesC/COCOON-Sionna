"""Sionna-RT integration for CSI and coverage extraction."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from .config import CoverageConfig, RadioConfig, SceneConfig, SolverConfig
from .logging_utils import progress_bar
from .mobility import Trajectory
from .sites import CandidateSite

logger = logging.getLogger(__name__)
_BACKEND_SELECTION: "BackendSelection | None" = None


def _complex_from_tuple(parts: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.asarray(parts[0]) + 1j * np.asarray(parts[1])


def _cfr_frequencies(radio: RadioConfig) -> np.ndarray:
    return np.asarray(radio.frequencies(), dtype=float)


def _noise_power_w(radio: RadioConfig) -> float:
    return float(Boltzmann * radio.temperature_k * radio.bandwidth_hz)


def _tx_power_w(power_dbm: float) -> float:
    return float(10 ** (power_dbm / 10.0) / 1000.0)


def _stack_padded(arrays: list[np.ndarray], fill_value: complex | float) -> np.ndarray:
    if not arrays:
        raise ValueError("arrays must not be empty")
    ndim = arrays[0].ndim
    if any(array.ndim != ndim for array in arrays):
        raise ValueError("all arrays must have the same rank")
    target_shape = tuple(max(array.shape[axis] for array in arrays) for axis in range(ndim))
    padded = []
    for array in arrays:
        pad_width = [(0, target_shape[axis] - array.shape[axis]) for axis in range(ndim)]
        if any(after > 0 for _, after in pad_width):
            array = np.pad(array, pad_width, mode="constant", constant_values=fill_value)
        padded.append(array)
    return np.stack(padded, axis=0)


@dataclass(slots=True, frozen=True)
class BackendSelection:
    device: str
    variant: str
    note: str | None = None


def _tail_text(text: str, lines: int = 8) -> str:
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(parts[-lines:])


def _parse_nvidia_compute_capabilities(text: str) -> tuple[float, ...]:
    capabilities: list[float] = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            capabilities.append(float(value))
        except ValueError:
            logger.debug("Ignoring unparseable NVIDIA compute capability entry: %s", value)
    return tuple(capabilities)


def _query_nvidia_compute_capabilities() -> tuple[float, ...] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        logger.debug("Unable to query NVIDIA compute capability: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug("nvidia-smi compute capability query failed: %s", _tail_text(result.stderr or result.stdout))
        return None

    capabilities = _parse_nvidia_compute_capabilities(result.stdout)
    return capabilities or None


def _zf_sinr_from_mimo_channel(
    channel: np.ndarray,
    total_tx_power_w: float,
    noise_power_w: float,
) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.complex128)
    if channel.ndim != 4:
        raise ValueError(f"Unsupported MIMO channel rank: {channel.ndim}")

    num_users, num_rx_ports, num_transmitters, num_tx_ports = channel.shape
    total_tx_ports = num_transmitters * num_tx_ports
    if num_users == 0 or total_tx_ports == 0:
        return np.zeros(num_users, dtype=float)

    effective_channel = np.zeros((num_users, total_tx_ports), dtype=np.complex128)
    for user_idx in range(num_users):
        user_channel = channel[user_idx].reshape(num_rx_ports, total_tx_ports)
        if not np.any(np.abs(user_channel) > 0.0):
            continue
        left_vectors, _, _ = np.linalg.svd(user_channel, full_matrices=False)
        combiner = left_vectors[:, 0]
        effective_channel[user_idx] = combiner.conj().T @ user_channel

    active_users = np.linalg.norm(effective_channel, axis=1) > 0.0
    if not np.any(active_users):
        return np.zeros(num_users, dtype=float)

    active_channel = effective_channel[active_users]
    precoder = np.linalg.pinv(active_channel, rcond=1e-9)
    column_norms = np.linalg.norm(precoder, axis=0)
    nonzero_columns = column_norms > 0.0
    precoder[:, nonzero_columns] /= column_norms[nonzero_columns]

    stream_power_w = total_tx_power_w / max(int(np.count_nonzero(active_users)), 1)
    gains = active_channel @ precoder
    total_received_power = stream_power_w * np.sum(np.abs(gains) ** 2, axis=1)
    desired_power = stream_power_w * np.abs(np.diag(gains)) ** 2
    interference_power = np.clip(total_received_power - desired_power, 0.0, None)
    active_sinr = desired_power / (interference_power + noise_power_w)

    sinr = np.zeros(num_users, dtype=float)
    sinr[active_users] = active_sinr.real
    return sinr


def _zf_sinr_from_paths(paths, total_tx_power_w: float, noise_power_w: float) -> np.ndarray:
    coeff = _complex_from_tuple(paths.a)
    if coeff.ndim != 5:
        raise ValueError(f"Unsupported path coefficient rank: {coeff.ndim}")
    narrowband_channel = np.sum(coeff, axis=-1)
    return _zf_sinr_from_mimo_channel(narrowband_channel, total_tx_power_w=total_tx_power_w, noise_power_w=noise_power_w)


def _probe_gpu_variant(variant: str) -> tuple[bool, str]:
    script = textwrap.dedent(
        f"""
        import mitsuba as mi
        mi.set_variant({variant!r})
        import sionna.rt as rt

        scene = rt.load_scene(rt.scene.etoile)
        scene.frequency = 3.5e9
        scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="VH",
        )
        scene.rx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="VH",
        )
        scene.add(rt.Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0), power_dbm=20.0))
        scene.add(rt.Receiver(name="rx", position=mi.Point3f(10.0, 0.0, 1.5)))
        solver = rt.PathSolver()
        solver(
            scene,
            max_depth=1,
            samples_per_src=16,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=1,
        )
        print("GPU_PROBE_OK")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"probe timed out after {exc.timeout}s"

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    details = _tail_text(combined) if combined else f"returncode={result.returncode}"
    success = result.returncode == 0 and "GPU_PROBE_OK" in (result.stdout or "")
    return success, details


def _detect_backend_selection() -> BackendSelection:
    global _BACKEND_SELECTION
    if _BACKEND_SELECTION is not None:
        return _BACKEND_SELECTION

    import mitsuba as mi

    available = set(mi.variants())
    fallback_note = "CPU fallback selected"
    if "cuda_ad_mono_polarized" in available:
        capabilities = _query_nvidia_compute_capabilities()
        if capabilities is not None and max(capabilities) < 7.0:
            fallback_note = (
                f"Detected NVIDIA GPU compute capability {max(capabilities):.1f}, "
                "below Mitsuba CUDA backend requirement sm_70"
            )
            logger.info("%s; falling back to CPU", fallback_note)
        else:
            logger.info("Probing GPU backend via Mitsuba variant cuda_ad_mono_polarized")
            ok, detail = _probe_gpu_variant("cuda_ad_mono_polarized")
            if ok:
                _BACKEND_SELECTION = BackendSelection(
                    device="GPU",
                    variant="cuda_ad_mono_polarized",
                    note="GPU probe succeeded",
                )
                logger.info("GPU backend probe succeeded")
                return _BACKEND_SELECTION
            fallback_note = detail
            if "requires .target sm_70 or higher" in detail:
                logger.info("GPU backend unsupported by detected NVIDIA GPU architecture; falling back to CPU. Probe tail: %s", detail)
            else:
                logger.warning("GPU backend probe failed; falling back to CPU. Probe tail: %s", detail)
    else:
        logger.info("CUDA Mitsuba variant unavailable; falling back to CPU")
        fallback_note = "CUDA Mitsuba variant unavailable"

    if "llvm_ad_mono_polarized" not in available:
        raise RuntimeError("No supported Mitsuba backend found; expected llvm_ad_mono_polarized")
    _BACKEND_SELECTION = BackendSelection(
        device="CPU",
        variant="llvm_ad_mono_polarized",
        note=fallback_note,
    )
    return _BACKEND_SELECTION


@dataclass(slots=True)
class SceneInputs:
    scene_path: Path | None
    metadata: dict[str, Any] | None


class SionnaRtRunner:
    def __init__(self, scene_cfg: SceneConfig, radio: RadioConfig, solver_cfg: SolverConfig, scene_inputs: SceneInputs) -> None:
        self.scene_cfg = scene_cfg
        self.radio = radio
        self.solver_cfg = solver_cfg
        self.scene_inputs = scene_inputs
        self._backend_logged = False
        self._backend_selection: BackendSelection | None = None
        self._drjit_runtime_configured = False
        self._reduced_export_labels: set[str] = set()

    def _uses_cpu_llvm_backend(self) -> bool:
        selection = self._backend_selection or _detect_backend_selection()
        return selection.device == "CPU" and selection.variant == "llvm_ad_mono_polarized"

    def _configure_drjit_runtime(self) -> None:
        if self._drjit_runtime_configured or not self._uses_cpu_llvm_backend():
            return
        import drjit as dr

        dr.set_flag(dr.JitFlag.MergeFunctions, False)
        dr.set_flag(dr.JitFlag.OptimizeCalls, False)
        logger.info("Configured conservative Dr.Jit LLVM flags for CPU backend stability")
        self._drjit_runtime_configured = True

    def _cleanup_drjit_runtime(self) -> None:
        if not self._uses_cpu_llvm_backend():
            return
        try:
            import drjit as dr

            dr.sync_thread()
            dr.flush_kernel_cache()
            dr.flush_malloc_cache()
            dr.kernel_history_clear()
        except Exception as exc:
            logger.debug("Dr.Jit runtime cleanup failed: %s", exc)
        gc.collect()

    def _should_export_full_csi(self, export_full: bool, label: str) -> bool:
        if not export_full:
            return False
        if self._uses_cpu_llvm_backend():
            if label not in self._reduced_export_labels:
                logger.info(
                    "Using reduced %s export on CPU LLVM backend for Dr.Jit stability; skipping raw CFR/CIR/tau tensors",
                    label,
                )
                self._reduced_export_labels.add(label)
            return False
        return True

    def _import_rt(self):
        import mitsuba as mi
        selection = self._backend_selection or _detect_backend_selection()
        if mi.variant() != selection.variant:
            mi.set_variant(selection.variant)
        self._backend_selection = selection
        if self.solver_cfg.require_gpu and selection.device != "GPU":
            raise RuntimeError(
                "GPU execution is required by configuration, but the Mitsuba CUDA backend is unavailable or unsupported "
                f"(selected backend: {selection.device} via {selection.variant}; note: {selection.note or 'n/a'})"
            )
        self._configure_drjit_runtime()
        if not self._backend_logged:
            logger.info("Using %s backend via Mitsuba variant %s", selection.device, selection.variant)
            self._backend_logged = True
        import sionna.rt as rt

        return {
            "mi": mi,
            "rt": rt,
            "PathSolver": rt.PathSolver,
            "PlanarArray": rt.PlanarArray,
            "RadioMapSolver": rt.RadioMapSolver,
            "Receiver": rt.Receiver,
            "Transmitter": rt.Transmitter,
            "load_scene": rt.load_scene,
        }

    def runtime_info(self) -> dict[str, str]:
        self._import_rt()
        assert self._backend_selection is not None
        return {
            "device": self._backend_selection.device,
            "variant": self._backend_selection.variant,
            "note": self._backend_selection.note or "",
        }

    def _configure_scene_arrays(self, scene, rt, tx_role: str, rx_role: str) -> None:
        if tx_role == "ap":
            tx_rows = self.radio.ap_num_rows
            tx_cols = self.radio.ap_num_cols
        elif tx_role == "ue":
            tx_rows = self.radio.ue_num_rows
            tx_cols = self.radio.ue_num_cols
        else:
            raise ValueError(f"Unsupported transmitter role: {tx_role}")

        if rx_role == "ap":
            rx_rows = self.radio.ap_num_rows
            rx_cols = self.radio.ap_num_cols
        elif rx_role == "ue":
            rx_rows = self.radio.ue_num_rows
            rx_cols = self.radio.ue_num_cols
        else:
            raise ValueError(f"Unsupported receiver role: {rx_role}")

        scene.tx_array = rt["PlanarArray"](
            num_rows=tx_rows,
            num_cols=tx_cols,
            vertical_spacing=self.radio.vertical_spacing,
            horizontal_spacing=self.radio.horizontal_spacing,
            pattern=self.radio.array_pattern,
            polarization=self.radio.array_polarization,
        )
        scene.rx_array = rt["PlanarArray"](
            num_rows=rx_rows,
            num_cols=rx_cols,
            vertical_spacing=self.radio.vertical_spacing,
            horizontal_spacing=self.radio.horizontal_spacing,
            pattern=self.radio.array_pattern,
            polarization=self.radio.array_polarization,
        )

    def _load_scene(self, tx_role: str = "ap", rx_role: str = "ue"):
        rt = self._import_rt()
        load_scene = rt["load_scene"]
        if self.scene_cfg.kind == "builtin":
            path = getattr(rt["rt"].scene, self.scene_cfg.sionna_scene)
            scene = load_scene(path)
        else:
            if self.scene_inputs.scene_path is None:
                raise ValueError("Scene path is required for non-builtin scenes")
            scene = load_scene(str(self.scene_inputs.scene_path))
        scene.frequency = self.radio.frequency_hz
        scene.bandwidth = self.radio.bandwidth_hz
        scene.temperature = self.radio.temperature_k
        self._configure_scene_arrays(scene, rt, tx_role=tx_role, rx_role=rx_role)
        return scene, rt

    def _orientation(self, mi, yaw_deg: float, pitch_deg: float):
        return mi.Point3f(float(np.radians(yaw_deg)), float(np.radians(pitch_deg)), 0.0)

    def _add_ap_transmitters(self, scene, rt, sites: list[CandidateSite]) -> list[str]:
        names: list[str] = []
        for site in sites:
            tx = rt["Transmitter"](
                name=f"ap_tx_{site.site_id}",
                position=rt["mi"].Point3f(float(site.x_m), float(site.y_m), float(site.z_m)),
                orientation=self._orientation(rt["mi"], site.yaw_deg, site.pitch_deg),
                power_dbm=self.radio.tx_power_dbm_ap,
            )
            scene.add(tx)
            names.append(tx.name)
        return names

    def _add_receivers(self, scene, rt, names: list[str], positions: np.ndarray, velocities: np.ndarray | None = None) -> None:
        for index, name in enumerate(names):
            velocity = velocities[index] if velocities is not None else np.zeros(3, dtype=float)
            rx = rt["Receiver"](
                name=name,
                position=rt["mi"].Point3f(*(float(v) for v in positions[index])),
                orientation=rt["mi"].Point3f(0.0, 0.0, 0.0),
                velocity=rt["mi"].Vector3f(*(float(v) for v in velocity)),
            )
            scene.add(rx)

    def _add_transmitters(self, scene, rt, names: list[str], positions: np.ndarray, velocities: np.ndarray, power_dbm: float) -> None:
        for index, name in enumerate(names):
            tx = rt["Transmitter"](
                name=name,
                position=rt["mi"].Point3f(*(float(v) for v in positions[index])),
                orientation=rt["mi"].Point3f(0.0, 0.0, 0.0),
                velocity=rt["mi"].Vector3f(*(float(v) for v in velocities[index])),
                power_dbm=power_dbm,
            )
            scene.add(tx)

    def _solve_paths(self, scene, rt):
        solver = rt["PathSolver"]()
        return solver(
            scene,
            max_depth=self.solver_cfg.path_max_depth,
            samples_per_src=self.solver_cfg.samples_per_src,
            synthetic_array=self.solver_cfg.synthetic_array,
            los=self.solver_cfg.los,
            specular_reflection=self.solver_cfg.specular_reflection,
            diffuse_reflection=self.solver_cfg.diffuse_reflection,
            refraction=self.solver_cfg.refraction,
            diffraction=self.solver_cfg.diffraction,
            edge_diffraction=self.solver_cfg.edge_diffraction,
            diffraction_lit_region=self.solver_cfg.diffraction_lit_region,
            seed=self.solver_cfg.seed,
        )

    def _link_power_from_cfr(self, cfr: np.ndarray) -> np.ndarray:
        if cfr.ndim == 6:
            axes = (1, 3, 4, 5)
        elif cfr.ndim == 5:
            axes = (1, 3, 4)
        else:
            raise ValueError(f"Unsupported CFR rank: {cfr.ndim}")
        return np.mean(np.abs(cfr) ** 2, axis=axes)

    def _link_power_from_paths(self, paths) -> np.ndarray:
        coeff = _complex_from_tuple(paths.a)
        if coeff.ndim != 5:
            raise ValueError(f"Unsupported path coefficient rank: {coeff.ndim}")
        return np.mean(np.sum(np.abs(coeff) ** 2, axis=-1), axis=(1, 3))

    def compute_radio_map(self, sites: list[CandidateSite], coverage: CoverageConfig) -> dict[str, Any]:
        logger.info("Computing radio map for %d AP transmitters", len(sites))
        try:
            scene, rt = self._load_scene(tx_role="ap", rx_role="ue")
            self._add_ap_transmitters(scene, rt, sites)
            solver = rt["RadioMapSolver"]()
            kwargs: dict[str, Any] = {
                "scene": scene,
                "cell_size": rt["mi"].Point2f(*coverage.cell_size_m),
                "samples_per_tx": self.solver_cfg.samples_per_tx,
                "max_depth": self.solver_cfg.radio_map_max_depth,
                "los": self.solver_cfg.los,
                "specular_reflection": self.solver_cfg.specular_reflection,
                "diffuse_reflection": self.solver_cfg.diffuse_reflection,
                "refraction": self.solver_cfg.refraction,
                "diffraction": self.solver_cfg.diffraction,
                "edge_diffraction": self.solver_cfg.edge_diffraction,
                "diffraction_lit_region": self.solver_cfg.diffraction_lit_region,
                "seed": self.solver_cfg.seed,
            }
            if coverage.center_m is not None and coverage.size_m is not None:
                kwargs["center"] = rt["mi"].Point3f(*coverage.center_m)
                kwargs["orientation"] = rt["mi"].Point3f(0.0, 0.0, 0.0)
                kwargs["size"] = rt["mi"].Point2f(*coverage.size_m)
            radio_map = solver(**kwargs)
            sinr = np.asarray(radio_map.sinr.numpy())
            rss = np.asarray(radio_map.rss.numpy())
            path_gain = np.asarray(radio_map.path_gain.numpy())
            cell_centers = np.asarray(radio_map.cell_centers.numpy())
            best_sinr = np.max(sinr, axis=0)
            return {
                "path_gain": path_gain,
                "rss": rss,
                "sinr": sinr,
                "best_sinr_db": 10.0 * np.log10(np.clip(best_sinr, 1e-12, None)),
                "cell_centers": cell_centers,
            }
        finally:
            self._cleanup_drjit_runtime()

    def compute_ap_ue_csi(self, sites: list[CandidateSite], trajectory: Trajectory, export_full: bool) -> dict[str, Any]:
        logger.info(
            "Computing AP-UE CSI for %d APs, %d UEs, %d snapshots using all-AP ZF distributed MIMO",
            len(sites),
            len(trajectory.ue_ids),
            len(trajectory.times_s),
        )
        try:
            export_full = self._should_export_full_csi(export_full, "AP-UE CSI")
            frequencies = _cfr_frequencies(self.radio)
            cfr_snapshots = []
            cir_snapshots = []
            tau_snapshots = []
            link_powers = []
            zf_sinr_snapshots = []
            total_tx_power_w = len(sites) * _tx_power_w(self.radio.tx_power_dbm_ap)
            noise_power_w = _noise_power_w(self.radio)
            with progress_bar(
                total=len(trajectory.times_s),
                desc="AP-UE CSI",
                unit="snapshot",
                leave=export_full,
            ) as progress:
                for t_idx in range(len(trajectory.times_s)):
                    scene, rt = self._load_scene(tx_role="ap", rx_role="ue")
                    self._add_ap_transmitters(scene, rt, sites)
                    rx_names = [f"ue_rx_{ue_id}" for ue_id in trajectory.ue_ids]
                    self._add_receivers(scene, rt, rx_names, trajectory.positions_m[t_idx], trajectory.velocities_mps[t_idx])
                    paths = self._solve_paths(scene, rt)
                    if export_full:
                        cfr = _complex_from_tuple(
                            paths.cfr(
                                frequencies=rt["mi"].Float(frequencies),
                                sampling_frequency=self.radio.effective_sampling_frequency_hz,
                                num_time_steps=1,
                                out_type="numpy",
                            )
                        )
                        cir, tau = paths.cir(
                            sampling_frequency=self.radio.effective_sampling_frequency_hz,
                            num_time_steps=1,
                            out_type="numpy",
                        )
                        cir_complex = _complex_from_tuple(cir)
                        cfr_snapshots.append(cfr)
                        cir_snapshots.append(cir_complex)
                        tau_snapshots.append(np.asarray(tau))
                    link_powers.append(self._link_power_from_paths(paths))
                    zf_sinr_snapshots.append(
                        _zf_sinr_from_paths(paths, total_tx_power_w=total_tx_power_w, noise_power_w=noise_power_w)
                    )
                    progress.update(1)
            power = np.stack(link_powers, axis=0) * _tx_power_w(self.radio.tx_power_dbm_ap)
            sinr = np.stack(zf_sinr_snapshots, axis=0)
            result = {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "best_sinr_db": 10.0 * np.log10(np.clip(sinr, 1e-12, None)),
                "link_power_w": power,
            }
            if export_full:
                result.update(
                    {
                        "cfr": _stack_padded(cfr_snapshots, fill_value=0.0 + 0.0j),
                        "cir": _stack_padded(cir_snapshots, fill_value=0.0 + 0.0j),
                        "tau": _stack_padded(tau_snapshots, fill_value=np.nan),
                    }
                )
            logger.info("AP-UE CSI computation complete")
            return result
        finally:
            self._cleanup_drjit_runtime()

    def compute_ap_ap_csi(self, sites: list[CandidateSite], export_full: bool) -> dict[str, Any]:
        logger.info("Computing AP-AP CSI for %d APs", len(sites))
        try:
            export_full = self._should_export_full_csi(export_full, "AP-AP CSI")
            frequencies = _cfr_frequencies(self.radio)
            scene, rt = self._load_scene(tx_role="ap", rx_role="ap")
            self._add_ap_transmitters(scene, rt, sites)
            rx_names = [f"ap_rx_{site.site_id}" for site in sites]
            positions = np.asarray([site.position for site in sites], dtype=float)
            velocities = np.zeros_like(positions)
            self._add_receivers(scene, rt, rx_names, positions, velocities)
            paths = self._solve_paths(scene, rt)
            if export_full:
                cfr = _complex_from_tuple(
                    paths.cfr(
                        frequencies=rt["mi"].Float(frequencies),
                        sampling_frequency=self.radio.effective_sampling_frequency_hz,
                        num_time_steps=1,
                        out_type="numpy",
                    )
                )
                cir, tau = paths.cir(
                    sampling_frequency=self.radio.effective_sampling_frequency_hz,
                    num_time_steps=1,
                    out_type="numpy",
                )
            power = self._link_power_from_paths(paths) * (10 ** (self.radio.tx_power_dbm_ap / 10.0) / 1000.0)
            np.fill_diagonal(power, 0.0)
            result = {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_site_ids": [site.site_id for site in sites],
                "link_power_w": power,
            }
            if export_full:
                result.update({"cfr": cfr, "cir": _complex_from_tuple(cir), "tau": np.asarray(tau)})
            logger.info("AP-AP CSI computation complete")
            return result
        finally:
            self._cleanup_drjit_runtime()

    def compute_ue_ue_csi(self, trajectory: Trajectory, export_full: bool) -> dict[str, Any]:
        logger.info(
            "Computing UE-UE CSI for %d UEs across %d snapshots",
            len(trajectory.ue_ids),
            len(trajectory.times_s),
        )
        try:
            export_full = self._should_export_full_csi(export_full, "UE-UE CSI")
            frequencies = _cfr_frequencies(self.radio)
            cfr_snapshots = []
            cir_snapshots = []
            tau_snapshots = []
            powers = []
            with progress_bar(
                total=len(trajectory.times_s),
                desc="UE-UE CSI",
                unit="snapshot",
                leave=export_full,
            ) as progress:
                for t_idx in range(len(trajectory.times_s)):
                    scene, rt = self._load_scene(tx_role="ue", rx_role="ue")
                    tx_names = [f"ue_tx_{ue_id}" for ue_id in trajectory.ue_ids]
                    rx_names = [f"ue_rx_{ue_id}" for ue_id in trajectory.ue_ids]
                    self._add_transmitters(
                        scene,
                        rt,
                        tx_names,
                        trajectory.positions_m[t_idx],
                        trajectory.velocities_mps[t_idx],
                        self.radio.tx_power_dbm_ue,
                    )
                    self._add_receivers(scene, rt, rx_names, trajectory.positions_m[t_idx], trajectory.velocities_mps[t_idx])
                    paths = self._solve_paths(scene, rt)
                    if export_full:
                        cfr = _complex_from_tuple(
                            paths.cfr(
                                frequencies=rt["mi"].Float(frequencies),
                                sampling_frequency=self.radio.effective_sampling_frequency_hz,
                                num_time_steps=1,
                                out_type="numpy",
                            )
                        )
                        cir, tau = paths.cir(
                            sampling_frequency=self.radio.effective_sampling_frequency_hz,
                            num_time_steps=1,
                            out_type="numpy",
                        )
                    power = self._link_power_from_paths(paths) * (10 ** (self.radio.tx_power_dbm_ue / 10.0) / 1000.0)
                    np.fill_diagonal(power, 0.0)
                    powers.append(power)
                    if export_full:
                        cfr_snapshots.append(cfr)
                        cir_snapshots.append(_complex_from_tuple(cir))
                        tau_snapshots.append(np.asarray(tau))
                    progress.update(1)
            link_power = np.stack(powers, axis=0)
            mean_peer_power = np.mean(np.where(link_power > 0.0, link_power, np.nan), axis=-1)
            mean_peer_power = np.nan_to_num(mean_peer_power, nan=0.0)
            need_weights = 1.0 / np.clip(mean_peer_power, 1e-12, None)
            need_weights /= max(float(np.mean(need_weights)), 1e-12)
            result = {
                "ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "link_power_w": link_power,
                "need_weights": need_weights,
            }
            if export_full:
                result.update(
                    {
                        "cfr": _stack_padded(cfr_snapshots, fill_value=0.0 + 0.0j),
                        "cir": _stack_padded(cir_snapshots, fill_value=0.0 + 0.0j),
                        "tau": _stack_padded(tau_snapshots, fill_value=np.nan),
                    }
                )
            logger.info("UE-UE CSI computation complete")
            return result
        finally:
            self._cleanup_drjit_runtime()


def load_scene_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
