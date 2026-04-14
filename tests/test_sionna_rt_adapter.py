import logging
import sys
from types import SimpleNamespace

import numpy as np

import cocoon_sionna.sionna_rt_adapter as adapter
from cocoon_sionna.sionna_rt_adapter import (
    _parse_nvidia_compute_capabilities,
    _stack_padded,
    _zf_sinr_from_mimo_channel,
    SionnaRtRunner,
    SceneInputs,
)
from cocoon_sionna.config import RadioConfig, SceneConfig, SolverConfig


def test_stack_padded_complex_arrays():
    first = np.ones((2, 3), dtype=np.complex64)
    second = 2.0 * np.ones((2, 5), dtype=np.complex64)

    stacked = _stack_padded([first, second], fill_value=0.0 + 0.0j)

    assert stacked.shape == (2, 2, 5)
    np.testing.assert_allclose(stacked[0, :, :3], first)
    np.testing.assert_allclose(stacked[0, :, 3:], 0.0)
    np.testing.assert_allclose(stacked[1], second)


def test_stack_padded_float_arrays_with_nan_fill():
    first = np.array([[1.0, 2.0, 3.0]])
    second = np.array([[4.0, 5.0]])

    stacked = _stack_padded([first, second], fill_value=np.nan)

    assert stacked.shape == (2, 1, 3)
    np.testing.assert_allclose(stacked[0], first)
    np.testing.assert_allclose(stacked[1, :, :2], second)
    assert np.isnan(stacked[1, 0, 2])


def test_zf_sinr_from_mimo_channel_cancels_interference_for_orthogonal_users():
    channel = np.zeros((2, 1, 2, 1), dtype=np.complex128)
    channel[0, 0, 0, 0] = 1.0
    channel[1, 0, 1, 0] = 1.0

    sinr = _zf_sinr_from_mimo_channel(channel, total_tx_power_w=2.0, noise_power_w=0.5)

    np.testing.assert_allclose(sinr, np.array([2.0, 2.0]))


def test_zf_sinr_from_mimo_channel_supports_receive_combining():
    channel = np.zeros((1, 2, 1, 2), dtype=np.complex128)
    channel[0, :, 0, :] = np.array([[1.0, 0.0], [0.0, 1.0]])

    sinr = _zf_sinr_from_mimo_channel(channel, total_tx_power_w=1.0, noise_power_w=0.25)

    np.testing.assert_allclose(sinr, np.array([4.0]))


def test_parse_nvidia_compute_capabilities_ignores_invalid_rows():
    parsed = _parse_nvidia_compute_capabilities("6.1\n7.5\nnot-a-number\n\n8.9\n")

    assert parsed == (6.1, 7.5, 8.9)


def test_detect_backend_selection_skips_gpu_probe_below_sm70(monkeypatch, caplog):
    fake_mitsuba = SimpleNamespace(variants=lambda: ["cuda_ad_mono_polarized", "llvm_ad_mono_polarized"])
    monkeypatch.setitem(sys.modules, "mitsuba", fake_mitsuba)
    monkeypatch.setattr(adapter, "_BACKEND_SELECTION", None)
    monkeypatch.setattr(adapter, "_query_nvidia_compute_capabilities", lambda: (6.1,))

    def _unexpected_probe(_variant: str):
        raise AssertionError("GPU probe should be skipped for compute capability below sm_70")

    monkeypatch.setattr(adapter, "_probe_gpu_variant", _unexpected_probe)

    with caplog.at_level(logging.INFO):
        selection = adapter._detect_backend_selection()

    assert selection.device == "CPU"
    assert selection.variant == "llvm_ad_mono_polarized"
    assert "6.1" in (selection.note or "")
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_cpu_backend_disables_full_csi_export(monkeypatch):
    runner = SionnaRtRunner(
        scene_cfg=SceneConfig(kind="builtin", sionna_scene="etoile"),
        radio=RadioConfig(),
        solver_cfg=SolverConfig(),
        scene_inputs=SceneInputs(scene_path=None, metadata=None),
    )
    monkeypatch.setattr(runner, "_uses_cpu_llvm_backend", lambda: True)

    assert runner._should_export_full_csi(True, "UE-UE CSI") is False
    assert runner._should_export_full_csi(False, "UE-UE CSI") is False


def test_runtime_info_raises_when_gpu_is_required(monkeypatch):
    runner = SionnaRtRunner(
        scene_cfg=SceneConfig(kind="builtin", sionna_scene="etoile"),
        radio=RadioConfig(),
        solver_cfg=SolverConfig(require_gpu=True),
        scene_inputs=SceneInputs(scene_path=None, metadata=None),
    )
    monkeypatch.setattr(
        adapter,
        "_detect_backend_selection",
        lambda: adapter.BackendSelection(device="CPU", variant="llvm_ad_mono_polarized", note="test fallback"),
    )

    try:
        runner.runtime_info()
    except RuntimeError as exc:
        assert "GPU execution is required" in str(exc)
    else:
        raise AssertionError("Expected runtime_info() to fail when GPU is required but unavailable")
