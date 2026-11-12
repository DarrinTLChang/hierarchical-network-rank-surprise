"""Opt-in consolidation of accepted HNRS output events, independent of truth.

This module does not run or modify RS detection, acceptance, references, or
classification. In particular, its regional outputs must not replace the native
unit-burst onsets passed to the network stage. Call it only after native detection
has completed, and save a distinct derived catalog rather than overwriting raw
catalogs. It is separate from the existing *unit-stage* posthoc-merge option.

Overlapping events or events separated by at most ``gap_ms`` form transitive
groups. A group's reported span is its earliest start through its latest end.
Consequently, positive gaps are bridged, not removed: that span can cover more
time than the union of its native intervals. Native member intervals, scores,
cores and contributor identities remain available without inventing a new RS
score or inferring new contributors from the enlarged reporting span.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan
from typing import Iterable, Literal

Grouping = Literal["regional", "network", "pooled_regional"]


@dataclass(frozen=True)
class NativeOutputEvent:
    """One accepted native call and its explicitly supplied source membership.

    IDs must be stable within a stream. ``stream_id`` separates hemispheres or
    other independently analyzed recordings. Contributor IDs/indices refer to
    native unit bursts, not planted events; this module never reads ground truth.
    The exporter is responsible for obtaining them from native detection cores.
    """

    native_event_id: str
    source_index: int
    start_ms: float
    end_ms: float
    stream_id: str
    region_labels: tuple[str, ...] = ()
    unit_ids: tuple[str, ...] = ()
    unit_burst_ids: tuple[str, ...] = ()
    contributing_indices: tuple[int, ...] = ()
    rs_score: float | None = None
    core_start_ms: float | None = None
    core_end_ms: float | None = None
    num_onsets: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.native_event_id, str) or not self.native_event_id:
            raise ValueError("native_event_id must be a nonempty string")
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("stream_id must explicitly identify the analysis stream")
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int) or self.source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if not isfinite(self.start_ms) or not isfinite(self.end_ms) or self.end_ms <= self.start_ms:
            raise ValueError("native interval must have finite start < end")
        for name in ("region_labels", "unit_ids", "unit_burst_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain nonempty strings")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        indices = tuple(self.contributing_indices)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices):
            raise ValueError("contributing_indices must contain nonnegative integers")
        object.__setattr__(self, "contributing_indices", tuple(sorted(set(indices))))
        if self.rs_score is not None and (isnan(self.rs_score) or self.rs_score < 0):
            raise ValueError("native RS score must be nonnegative and not NaN")
        if (self.core_start_ms is None) != (self.core_end_ms is None):
            raise ValueError("provide both native core boundaries or neither")
        if self.core_start_ms is not None:
            if not isfinite(self.core_start_ms) or not isfinite(self.core_end_ms) or self.core_end_ms < self.core_start_ms:
                raise ValueError("native core must have finite start <= end")
        if self.num_onsets is not None and (
            isinstance(self.num_onsets, bool) or not isinstance(self.num_onsets, int) or self.num_onsets < 0
        ):
            raise ValueError("num_onsets must be a nonnegative integer when provided")


@dataclass(frozen=True)
class MergedOutputEvent:
    """A reported interval with lossless references to its native members."""

    merged_event_id: str
    grouping: Grouping
    stream_id: str
    start_ms: float
    end_ms: float
    gap_ms: float
    members: tuple[NativeOutputEvent, ...]

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def native_event_ids(self) -> tuple[str, ...]:
        return tuple(member.native_event_id for member in self.members)

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(member.source_index for member in self.members)

    @property
    def native_intervals(self) -> tuple[tuple[float, float], ...]:
        return tuple((member.start_ms, member.end_ms) for member in self.members)

    @property
    def native_rs_scores(self) -> tuple[float | None, ...]:
        # A merged reporting interval has no newly evaluated RS significance.
        return tuple(member.rs_score for member in self.members)

    @property
    def native_core_intervals(self) -> tuple[tuple[float | None, float | None], ...]:
        return tuple((member.core_start_ms, member.core_end_ms) for member in self.members)

    @property
    def region_labels(self) -> tuple[str, ...]:
        return tuple(sorted({value for member in self.members for value in member.region_labels}))

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value for member in self.members for value in member.unit_ids}))

    @property
    def unit_burst_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value for member in self.members for value in member.unit_burst_ids}))

    @property
    def contributing_indices(self) -> tuple[int, ...]:
        return tuple(sorted({value for member in self.members for value in member.contributing_indices}))

    @property
    def native_covered_duration_ms(self) -> float:
        """Duration of the union of native intervals, excluding bridged gaps."""
        total = 0.0
        start, end = self.members[0].start_ms, self.members[0].end_ms
        for member in self.members[1:]:
            if member.start_ms <= end:
                end = max(end, member.end_ms)
            else:
                total += end - start
                start, end = member.start_ms, member.end_ms
        return total + end - start

    @property
    def bridged_gap_duration_ms(self) -> float:
        return self.duration_ms - self.native_covered_duration_ms


def merge_output_events(
    events: Iterable[NativeOutputEvent],
    *,
    gap_ms: float = 25.0,
    grouping: Grouping,
) -> tuple[MergedOutputEvent, ...]:
    """Merge accepted output intervals without changing classification rules.

    ``regional`` keeps every region/stream separate. ``network`` merges a final
    network-stage stream. ``pooled_regional`` makes a distinctly labeled derived
    stream pooled across regions; it is not a replacement for native stage3.
    All modes keep different ``stream_id`` values separate. Membership is only
    the union of supplied member identities, never an overlap-based reassignment.
    The maximum end of the complete current group implements transitive merging.
    No floating-point tolerance is added to the requested inclusive gap bound.
    """
    if grouping not in ("regional", "network", "pooled_regional"):
        raise ValueError(f"unknown grouping: {grouping!r}")
    if not isfinite(gap_ms) or gap_ms < 0:
        raise ValueError("gap_ms must be finite and nonnegative")
    copied = tuple(events)
    if any(not isinstance(event, NativeOutputEvent) for event in copied):
        raise TypeError("events must contain NativeOutputEvent records")
    identities = [(event.stream_id, event.native_event_id) for event in copied]
    if len(identities) != len(set(identities)):
        raise ValueError("native_event_id must be unique within each stream")
    groups: dict[tuple[str, ...], list[NativeOutputEvent]] = {}
    for event in copied:
        if grouping == "regional":
            if len(event.region_labels) != 1:
                raise ValueError("regional grouping requires exactly one region label per native call")
            key = (event.stream_id, event.region_labels[0])
        else:
            key = (event.stream_id,)
        groups.setdefault(key, []).append(event)

    member_groups: list[tuple[NativeOutputEvent, ...]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda event: (
            event.start_ms, event.end_ms, event.source_index, event.native_event_id
        ))
        members = [ordered[0]]
        maximum_end = ordered[0].end_ms
        for event in ordered[1:]:
            if event.start_ms <= maximum_end + gap_ms:
                members.append(event)
                maximum_end = max(maximum_end, event.end_ms)
            else:
                member_groups.append(tuple(members))
                members = [event]
                maximum_end = event.end_ms
        member_groups.append(tuple(members))
    member_groups.sort(key=lambda members: (
        members[0].stream_id, members[0].start_ms,
        max(event.end_ms for event in members), tuple(event.native_event_id for event in members)
    ))
    return tuple(MergedOutputEvent(
        merged_event_id=f"{grouping}:{index:06d}", grouping=grouping,
        stream_id=members[0].stream_id, start_ms=members[0].start_ms,
        end_ms=max(event.end_ms for event in members), gap_ms=float(gap_ms), members=members,
    ) for index, members in enumerate(member_groups))
