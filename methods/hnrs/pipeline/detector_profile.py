from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

import config as cfg


WIN_SHUFF_IMPLEMENTATION_VERSION = "b1-v2"
DETECTOR_IMPLEMENTATION_VERSION = "rs-v3"


PROFILE_GLOBAL_NAMES = (
    "maxisi_toggle",
    "alpha_meanisi_toggle",
    "rankSurprise_toggle",
    "pooling_toggle",
    "ibi_merge_factor",
    "RS_region_burst_toggle",
    "RS_NETWORK_ONSETS_TOGGLE",
    "RS_alpha_percentage_stage1",
    "RS_alpha_stage1",
    "RS_Limit_stage1",
    "RS_Percentile_Limit_stage1",
    "RS_alpha_percentage_region",
    "RS_alpha_region",
    "RS_Limit_region",
    "RS_Percentile_Limit_region",
    "RS_alpha_percentage_network",
    "RS_alpha_network",
    "RS_Limit_network",
    "RS_Percentile_Limit_network",
    "MIN_SPIKES_IN_BURST",
    "RS_WIN_SHUFF_STAGE1_ENABLE",
    "RS_WIN_SHUFF_AUTO_FROM_RECORDING",
    "RS_WIN_SHUFF_WINDOW_MS",
    "RS_WIN_SHUFF_BIN_MS",
    "RS_WIN_SHUFF_SEED",
    "RS_STAGE1_LOCAL_SEGMENT_ENABLE",
    "RS_STAGE1_SEGMENT_MODE",
    "RS_STAGE1_SEGMENT_LEN_S",
    "RS_STAGE1_SEGMENT_OVERLAP_FRACTION",
    "RS_STAGE1_SEGMENT_MIN_SPIKES",
    "RS_STAGE1_CUSTOM_WINDOWS_S",
    "RS_STAGE1_POSTHOC_MERGE_ENABLE",
    "RS_STAGE1_POSTHOC_MERGE_GAP_MS",
    "RS_STAGE1_POSTHOC_DEDUP_IOU_MIN",
    "RS_OFFSET_NULL_ENABLE",
    "RS_OFFSET_NULL_SEED",
    "RS_OFFSET_NULL_MAX_OFFSET_MS",
    "MIN_UNIT_BURST_DURATION_MS",
    "MAX_UNIT_BURST_DURATION_MS",
    "MIN_REGION_BURST_SUPPORT_DURATION_MS",
    "MIN_NETWORK_BURST_SUPPORT_DURATION_MS",
    "MAX_NETWORK_BURST_SUPPORT_DURATION_MS",
    "MIN_BURST_DURATION",
    "MAX_BURST_DURATION",
    "MIN_UNIQUE_CHANNELS_REGION",
    "MIN_UNIQUE_CHANNELS_NETWORK",
    "MIN_UNIQUE_REGIONS_NETWORK",
    "REGION_SPAN_ONSET_PLUS_LENGTH",
    "NETWORK_SPAN_ONSET_PLUS_LENGTH",
    "APPLY_MIN_BURST_DURATION_ALL_STAGES",
    "FR_MIN_HZ",
    "SNR_MIN",
    "SNR_MAX",
)


@dataclass(frozen=True)
class DetectorProfile:
    profile_name: str
    profile_version: int
    run_stage1: bool
    run_region_stage: bool
    run_network_stage: bool
    stage1_alpha: float
    region_alpha: float | None
    network_alpha: float | None
    stage1_limit: float | None
    region_limit: float | None
    network_limit: float | None
    stage1_percentile_limit: int
    region_percentile_limit: int | None
    network_percentile_limit: int | None
    min_spikes_in_burst: int
    win_shuff_enabled: bool
    win_shuff_auto_from_recording: bool
    win_shuff_window_ms: float | None
    win_shuff_bin_ms: float | None
    win_shuff_seed: int
    local_segmentation_enabled: bool
    offset_null_enabled: bool
    offset_null_seed: int
    offset_null_max_offset_ms: float | None
    min_unit_burst_duration_ms: float
    max_unit_burst_duration_ms: float | None
    min_region_support_duration_ms: float | None
    min_network_support_duration_ms: float | None
    max_network_support_duration_ms: float | None
    min_unique_channels_region: int | None
    min_unique_channels_network: int | None
    min_unique_regions_network: int | None
    region_span_onset_plus_length: bool | None
    network_span_onset_plus_length: bool | None
    apply_unit_quality_filters: bool
    pooling_enabled: bool
    adaptive_threshold_enabled: bool
    ibi_merge_factor: float
    fr_min_hz: float | None
    snr_min: float | None
    snr_max: float | None
    segment_mode: str = "nonoverlap"
    segment_len_s: float = 5.0
    segment_overlap_fraction: float = 0.0
    segment_min_spikes: int = 3
    custom_windows_s: tuple[tuple[float, float], ...] = ()
    posthoc_merge_enabled: bool = False
    posthoc_merge_gap_ms: float = 0.0
    posthoc_dedup_iou_min: float = 0.8
    win_shuff_implementation_version: str = WIN_SHUFF_IMPLEMENTATION_VERSION
    detector_implementation_version: str = DETECTOR_IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        # Canonicalize inactive custom settings before serialization or hashing.
        # This also applies to profiles made with dataclasses.replace().
        if self.win_shuff_enabled and self.win_shuff_auto_from_recording:
            object.__setattr__(self, "win_shuff_window_ms", None)
            object.__setattr__(self, "win_shuff_bin_ms", None)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if not self.win_shuff_enabled:
            # Inactive shuffle settings do not affect classic-RS identity.
            values.pop("win_shuff_implementation_version")
        return json.loads(json.dumps(values, allow_nan=False))

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        return (
            f"{self.profile_name}-p{self.profile_version}-"
            f"{self.digest[:12]}"
        )


TEST1_PROFILE = DetectorProfile(
    profile_name="test1_network_v1",
    profile_version=1,
    run_stage1=True,
    run_region_stage=True,
    run_network_stage=True,
    stage1_alpha=0.08,
    region_alpha=0.06,
    network_alpha=0.03,
    stage1_limit=None,
    region_limit=None,
    network_limit=None,
    stage1_percentile_limit=75,
    region_percentile_limit=75,
    network_percentile_limit=75,
    min_spikes_in_burst=3,
    win_shuff_enabled=False,
    win_shuff_auto_from_recording=False,
    win_shuff_window_ms=200.0,
    win_shuff_bin_ms=10.0,
    win_shuff_seed=123,
    local_segmentation_enabled=False,
    offset_null_enabled=True,
    offset_null_seed=123,
    offset_null_max_offset_ms=10000.0,
    min_unit_burst_duration_ms=20.0,
    max_unit_burst_duration_ms=None,
    min_region_support_duration_ms=20.0,
    min_network_support_duration_ms=20.0,
    max_network_support_duration_ms=10000.0,
    min_unique_channels_region=1,
    min_unique_channels_network=1,
    min_unique_regions_network=2,
    region_span_onset_plus_length=True,
    network_span_onset_plus_length=True,
    apply_unit_quality_filters=True,
    pooling_enabled=False,
    adaptive_threshold_enabled=False,
    ibi_merge_factor=0.0,
    fr_min_hz=0.0,
    snr_min=0.8,
    snr_max=25.0,
)


BURSTPAUSE_PROFILE_BASE = DetectorProfile(
    profile_name="burstpause_stage1_v1",
    profile_version=1,
    run_stage1=True,
    run_region_stage=False,
    run_network_stage=False,
    stage1_alpha=0.08,
    region_alpha=None,
    network_alpha=None,
    stage1_limit=None,
    region_limit=None,
    network_limit=None,
    stage1_percentile_limit=75,
    region_percentile_limit=None,
    network_percentile_limit=None,
    min_spikes_in_burst=3,
    win_shuff_enabled=True,
    win_shuff_auto_from_recording=False,
    win_shuff_window_ms=0.0,
    win_shuff_bin_ms=0.0,
    win_shuff_seed=123,
    local_segmentation_enabled=False,
    offset_null_enabled=False,
    offset_null_seed=123,
    offset_null_max_offset_ms=None,
    min_unit_burst_duration_ms=0.0,
    max_unit_burst_duration_ms=None,
    min_region_support_duration_ms=None,
    min_network_support_duration_ms=None,
    max_network_support_duration_ms=None,
    min_unique_channels_region=None,
    min_unique_channels_network=None,
    min_unique_regions_network=None,
    region_span_onset_plus_length=None,
    network_span_onset_plus_length=None,
    apply_unit_quality_filters=False,
    pooling_enabled=False,
    adaptive_threshold_enabled=False,
    ibi_merge_factor=0.0,
    fr_min_hz=None,
    snr_min=None,
    snr_max=None,
)


def burstpause_profile(
    window_ms: float | None = None,
    bin_ms: float | None = None,
    *,
    auto_from_recording: bool = False,
) -> DetectorProfile:
    if auto_from_recording:
        effective_window, effective_bin = None, None
    else:
        effective_window, effective_bin = resolve_win_shuff_params_ms(
            None, auto_from_recording=False, window_ms=window_ms, bin_ms=bin_ms
        )
    return replace(
        BURSTPAUSE_PROFILE_BASE,
        win_shuff_auto_from_recording=bool(auto_from_recording),
        win_shuff_window_ms=effective_window,
        win_shuff_bin_ms=effective_bin,
    )


def resolve_win_shuff_params_ms(
    record_len_s: float | None,
    *,
    auto_from_recording: bool,
    window_ms: float | None = None,
    bin_ms: float | None = None,
) -> tuple[float, float]:
    """Resolve only the selected policy; never inspect inactive settings.

    Automatic sizing is exactly 10% of the analyzed interval, followed by
    10% of that window. Custom values remain unchanged and must contain an
    integer bin count within max(1e-9, 8 ULPs) of the window/bin ratio.
    """
    if bool(auto_from_recording):
        try:
            record_ms = float(record_len_s) * 1000.0
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Automatic WIN-SHUFF requires a finite positive analyzed duration") from exc
        resolved_window = record_ms / 10.0
        resolved_bin = record_ms / 100.0
        if not all(math.isfinite(value) and value > 0 for value in
                   (record_ms, resolved_window, resolved_bin)):
            raise ValueError("Automatic WIN-SHUFF requires a finite positive, representable analyzed duration")
        return resolved_window, resolved_bin

    message = (
        f"Custom WIN-SHUFF window={window_ms!r} ms, bin={bin_ms!r} ms: "
        "both must be finite and positive, and window/bin must be a positive integer"
    )
    try:
        resolved_window, resolved_bin = float(window_ms), float(bin_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not all(math.isfinite(value) and value > 0 for value in
               (resolved_window, resolved_bin)):
        raise ValueError(message)
    ratio = resolved_window / resolved_bin
    if not math.isfinite(ratio):
        raise ValueError(message)
    n_bins = round(ratio)
    tolerance = max(1e-9, 8.0 * math.ulp(ratio))
    if n_bins < 1 or abs(ratio - n_bins) > tolerance:
        raise ValueError(message)
    return resolved_window, resolved_bin


def normal_rs_profile(
    *,
    alphas: tuple[float, float, float],
    win_shuff: tuple[float | None, float | None] | None = None,
    win_shuff_auto: bool | None = None,
) -> DetectorProfile:
    stage1_alpha, region_alpha, network_alpha = map(float, alphas)
    auto = (bool(getattr(cfg, "RS_WIN_SHUFF_AUTO_FROM_RECORDING", False))
            if win_shuff_auto is None else bool(win_shuff_auto))
    enabled = bool(cfg.RS_WIN_SHUFF_STAGE1_ENABLE)
    if not enabled or auto:
        window_ms, bin_ms = None, None
    else:
        requested = win_shuff if win_shuff is not None else (None, None)
        window_ms, bin_ms = resolve_win_shuff_params_ms(
            None, auto_from_recording=False,
            window_ms=requested[0], bin_ms=requested[1],
        )
    custom_windows = tuple(
        (float(start), float(end))
        for start, end in getattr(cfg, "RS_STAGE1_CUSTOM_WINDOWS_S", [])
    )
    return DetectorProfile(
        profile_name="normal_rs",
        profile_version=1,
        run_stage1=True,
        run_region_stage=bool(getattr(cfg, "RS_region_burst_toggle", True)),
        run_network_stage=bool(getattr(cfg, "RS_NETWORK_ONSETS_TOGGLE", True)),
        stage1_alpha=stage1_alpha,
        region_alpha=region_alpha,
        network_alpha=network_alpha,
        stage1_limit=getattr(cfg, "RS_Limit_stage1", None),
        region_limit=getattr(cfg, "RS_Limit_region", None),
        network_limit=getattr(cfg, "RS_Limit_network", None),
        stage1_percentile_limit=int(cfg.RS_Percentile_Limit_stage1),
        region_percentile_limit=int(cfg.RS_Percentile_Limit_region),
        network_percentile_limit=int(cfg.RS_Percentile_Limit_network),
        min_spikes_in_burst=int(cfg.MIN_SPIKES_IN_BURST),
        win_shuff_enabled=enabled,
        win_shuff_auto_from_recording=auto,
        win_shuff_window_ms=window_ms,
        win_shuff_bin_ms=bin_ms,
        win_shuff_seed=int(cfg.RS_WIN_SHUFF_SEED),
        local_segmentation_enabled=bool(cfg.RS_STAGE1_LOCAL_SEGMENT_ENABLE),
        offset_null_enabled=bool(cfg.RS_OFFSET_NULL_ENABLE),
        offset_null_seed=int(cfg.RS_OFFSET_NULL_SEED),
        offset_null_max_offset_ms=cfg.RS_OFFSET_NULL_MAX_OFFSET_MS,
        min_unit_burst_duration_ms=float(
            getattr(cfg, "MIN_UNIT_BURST_DURATION_MS", cfg.MIN_BURST_DURATION)
        ),
        max_unit_burst_duration_ms=getattr(
            cfg, "MAX_UNIT_BURST_DURATION_MS", None
        ),
        min_region_support_duration_ms=float(
            getattr(
                cfg,
                "MIN_REGION_BURST_SUPPORT_DURATION_MS",
                cfg.MIN_BURST_DURATION,
            )
        ),
        min_network_support_duration_ms=float(
            getattr(
                cfg,
                "MIN_NETWORK_BURST_SUPPORT_DURATION_MS",
                cfg.MIN_BURST_DURATION,
            )
        ),
        max_network_support_duration_ms=getattr(
            cfg,
            "MAX_NETWORK_BURST_SUPPORT_DURATION_MS",
            cfg.MAX_BURST_DURATION,
        ),
        min_unique_channels_region=int(cfg.MIN_UNIQUE_CHANNELS_REGION),
        min_unique_channels_network=int(cfg.MIN_UNIQUE_CHANNELS_NETWORK),
        min_unique_regions_network=int(cfg.MIN_UNIQUE_REGIONS_NETWORK),
        region_span_onset_plus_length=bool(cfg.REGION_SPAN_ONSET_PLUS_LENGTH),
        network_span_onset_plus_length=bool(cfg.NETWORK_SPAN_ONSET_PLUS_LENGTH),
        apply_unit_quality_filters=True,
        pooling_enabled=bool(getattr(cfg, "pooling_toggle", False)),
        adaptive_threshold_enabled=bool(
            getattr(cfg, "alpha_meanisi_toggle", False)
        ),
        ibi_merge_factor=float(getattr(cfg, "ibi_merge_factor", 0.0)),
        fr_min_hz=float(cfg.FR_MIN_HZ),
        snr_min=float(cfg.SNR_MIN),
        snr_max=None if cfg.SNR_MAX is None else float(cfg.SNR_MAX),
        segment_mode=str(cfg.RS_STAGE1_SEGMENT_MODE),
        segment_len_s=float(cfg.RS_STAGE1_SEGMENT_LEN_S),
        segment_overlap_fraction=float(
            cfg.RS_STAGE1_SEGMENT_OVERLAP_FRACTION
        ),
        segment_min_spikes=int(cfg.RS_STAGE1_SEGMENT_MIN_SPIKES),
        custom_windows_s=custom_windows,
        posthoc_merge_enabled=bool(cfg.RS_STAGE1_POSTHOC_MERGE_ENABLE),
        posthoc_merge_gap_ms=float(cfg.RS_STAGE1_POSTHOC_MERGE_GAP_MS),
        posthoc_dedup_iou_min=float(cfg.RS_STAGE1_POSTHOC_DEDUP_IOU_MIN),
    )


def apply_detector_profile(profile: DetectorProfile) -> None:
    import pipeline.detection as detection
    import pipeline.stats as stats
    import pipeline.utils as utils

    modules = (cfg, detection, utils, stats)
    values = {
        "maxisi_toggle": False,
        "alpha_meanisi_toggle": bool(profile.adaptive_threshold_enabled),
        "rankSurprise_toggle": bool(profile.run_stage1),
        "pooling_toggle": bool(profile.pooling_enabled),
        "ibi_merge_factor": float(profile.ibi_merge_factor),
        "RS_region_burst_toggle": bool(profile.run_region_stage),
        "RS_NETWORK_ONSETS_TOGGLE": bool(profile.run_network_stage),
        "RS_alpha_percentage_stage1": float(profile.stage1_alpha),
        "RS_alpha_stage1": -np.log(float(profile.stage1_alpha)),
        "RS_Limit_stage1": profile.stage1_limit,
        "RS_Percentile_Limit_stage1": int(profile.stage1_percentile_limit),
        "MIN_SPIKES_IN_BURST": int(profile.min_spikes_in_burst),
        "RS_WIN_SHUFF_STAGE1_ENABLE": bool(profile.win_shuff_enabled),
        "RS_WIN_SHUFF_AUTO_FROM_RECORDING": bool(
            profile.win_shuff_auto_from_recording
        ),
        "RS_WIN_SHUFF_SEED": int(profile.win_shuff_seed),
        "RS_STAGE1_LOCAL_SEGMENT_ENABLE": bool(
            profile.local_segmentation_enabled
        ),
        "RS_STAGE1_SEGMENT_MODE": str(profile.segment_mode),
        "RS_STAGE1_SEGMENT_LEN_S": float(profile.segment_len_s),
        "RS_STAGE1_SEGMENT_OVERLAP_FRACTION": float(
            profile.segment_overlap_fraction
        ),
        "RS_STAGE1_SEGMENT_MIN_SPIKES": int(profile.segment_min_spikes),
        "RS_STAGE1_CUSTOM_WINDOWS_S": [
            tuple(window) for window in profile.custom_windows_s
        ],
        "RS_STAGE1_POSTHOC_MERGE_ENABLE": bool(
            profile.posthoc_merge_enabled
        ),
        "RS_STAGE1_POSTHOC_MERGE_GAP_MS": float(
            profile.posthoc_merge_gap_ms
        ),
        "RS_STAGE1_POSTHOC_DEDUP_IOU_MIN": float(
            profile.posthoc_dedup_iou_min
        ),
        "RS_OFFSET_NULL_ENABLE": bool(profile.offset_null_enabled),
        "RS_OFFSET_NULL_SEED": int(profile.offset_null_seed),
        "RS_OFFSET_NULL_MAX_OFFSET_MS": profile.offset_null_max_offset_ms,
        "MIN_UNIT_BURST_DURATION_MS": float(
            profile.min_unit_burst_duration_ms
        ),
        "MAX_UNIT_BURST_DURATION_MS": profile.max_unit_burst_duration_ms,
        "APPLY_MIN_BURST_DURATION_ALL_STAGES": False,
        "FR_MIN_HZ": (
            0.0 if profile.fr_min_hz is None else float(profile.fr_min_hz)
        ),
    }
    if profile.win_shuff_enabled and not profile.win_shuff_auto_from_recording:
        window_ms, bin_ms = resolve_win_shuff_params_ms(
            None, auto_from_recording=False,
            window_ms=profile.win_shuff_window_ms, bin_ms=profile.win_shuff_bin_ms,
        )
        values.update(RS_WIN_SHUFF_WINDOW_MS=window_ms, RS_WIN_SHUFF_BIN_MS=bin_ms)
    if profile.region_alpha is not None:
        values.update(
            {
                "RS_alpha_percentage_region": float(profile.region_alpha),
                "RS_alpha_region": -np.log(float(profile.region_alpha)),
                "RS_Limit_region": profile.region_limit,
                "RS_Percentile_Limit_region": int(
                    profile.region_percentile_limit
                ),
            }
        )
    if profile.network_alpha is not None:
        values.update(
            {
                "RS_alpha_percentage_network": float(profile.network_alpha),
                "RS_alpha_network": -np.log(float(profile.network_alpha)),
                "RS_Limit_network": profile.network_limit,
                "RS_Percentile_Limit_network": int(
                    profile.network_percentile_limit
                ),
            }
        )
    optional_values = {
        "MIN_REGION_BURST_SUPPORT_DURATION_MS": (
            profile.min_region_support_duration_ms
        ),
        "MIN_NETWORK_BURST_SUPPORT_DURATION_MS": (
            profile.min_network_support_duration_ms
        ),
        "MAX_NETWORK_BURST_SUPPORT_DURATION_MS": (
            profile.max_network_support_duration_ms
        ),
        "MIN_UNIQUE_CHANNELS_REGION": profile.min_unique_channels_region,
        "MIN_UNIQUE_CHANNELS_NETWORK": profile.min_unique_channels_network,
        "MIN_UNIQUE_REGIONS_NETWORK": profile.min_unique_regions_network,
        "REGION_SPAN_ONSET_PLUS_LENGTH": profile.region_span_onset_plus_length,
        "NETWORK_SPAN_ONSET_PLUS_LENGTH": profile.network_span_onset_plus_length,
    }
    values.update(optional_values)
    values["MIN_BURST_DURATION"] = float(
        profile.min_unit_burst_duration_ms
        if profile.min_network_support_duration_ms is None
        else profile.min_network_support_duration_ms
    )
    values["MAX_BURST_DURATION"] = profile.max_network_support_duration_ms
    values["SNR_MIN"] = (
        0.0 if profile.snr_min is None else float(profile.snr_min)
    )
    values["SNR_MAX"] = (
        None if profile.snr_max is None else float(profile.snr_max)
    )

    for module in modules:
        for name, value in values.items():
            setattr(module, name, value)


@contextmanager
def detector_profile_context(profile: DetectorProfile):
    """Apply a profile temporarily and restore all copied config globals."""
    import pipeline.detection as detection
    import pipeline.stats as stats
    import pipeline.utils as utils

    modules = (cfg, detection, utils, stats)
    missing = object()
    saved = {
        module: {
            name: getattr(module, name, missing) for name in PROFILE_GLOBAL_NAMES
        }
        for module in modules
    }
    try:
        apply_detector_profile(profile)
        yield profile
    finally:
        for module, values in saved.items():
            for name, value in values.items():
                if value is missing:
                    if hasattr(module, name):
                        delattr(module, name)
                else:
                    setattr(module, name, value)
