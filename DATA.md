# Fixed synthetic datasets and ground truth

The authoritative inventory is `datasets/recording_manifest.csv`; detailed
schemas and indexing conventions are in `datasets/data_dictionary.json`.
Paths are repository-root-relative. All 303 MAT inputs are synthetic, not
patient-derived surrogates. Dataset generators are deliberately not released.

## Cohorts

| Manifest cohort | Inputs | Role |
| --- | ---: | --- |
| `burst_pause` | 50 | Unit-only RS/HNRS comparison, 30 seconds each |
| `network_development` | 30 | Parameter development, 250 seconds each |
| `network_burst` | 60 | Shared general-detection/network-classification evaluation |
| `network_null` | 42 | 21 background/independently timed regional-burst pairs |
| `network_shift` | 120 | Two fixed shifted realizations per evaluation recording |
| `synthetic_example` | 1 | Separate homogeneous single-unit illustration |

### Shared multiregion inputs

Each development/evaluation input contains 75 active units across five
simulated groups, 12 single-region and six multiregion population targets,
and six isolated single-unit burst challenges. Development recording IDs are
961001-961030; evaluation IDs are 73001-73060. Both main benchmark tasks use
the same 60 evaluation spike realizations, not separate cohorts.

Trial directories contain the exact spike input and available channel maps,
population targets, regional components, unit targets, and planted-spike truth.
Use `event_manifest.csv`'s `actual_start_ms`/`actual_end_ms` for unpadded
population scoring. Historical padded `score_*` columns are not the scoring
windows for these results. Some original metadata descriptors precede the
isolated-unit additions; cohort-level challenge and added-spike tables identify
those additions without changing the 18 population-target labels.

### Burst-Pause

All spikes belong to planted bursts; this is not a tonic-background null test.
The 50 inputs contain 2,811 bursts and 33,478 spikes. Unit identity was fixed,
not selected by detector performance. Inherited `calibration/` and
`evaluation/` storage names identify source collections; **all 50 inputs are
evaluated together**. They are not a training/test split for this comparison.
The separate illustration does not belong to the 50-input benchmark.

### Null and shifted inputs

All null source IDs 1-21 are included. Background-only recordings have no
planted bursts. Their partners use the same backgrounds with 32 independently
timed regional burst packets. Neither condition has planted network bursts;
chance cross-region coincidences are allowed. Header-only network-truth
tables correctly indicate no planted network targets, not no coincidences.

Shift inputs are the actual frozen realizations used by the controlled
desynchronization analysis. Their offsets were constrained to separate planted
regional components; this is not an unconstrained false-positive null.
The offset manifest records the choices. The readout concerns detection-count
responses, not recovery against the original unshifted network intervals.

## Format and units

`spikeTime.mat` stores a MATLAB `spikeTime` struct with `electrode`, `time`,
`snr`, and `dataSegmentLength` fields. Spike times are in **milliseconds**;
`dataSegmentLength` is in **seconds**. Empty channel/cluster slots can occur.
The multiregion struct has 50 electrode slots and 75 active units; use the
channel map rather than interpreting slot count as active-unit count.

Anatomical-looking channel names are simulator labels, not patient recording
sites. Check the dictionary before interchanging zero-based network cluster
indices with the one-based Burst-Pause `Cluster` convention.

## Current references

`results/reported_settings.json` records HNRS 1%/3%/15%, logIBEI 100 ms,
Network-CMA 0.75, and meaRtools Sigma 45. HNRS uses three primary reference-seed
analyses; blank comparator seed fields denote deterministic runs.

`results/network_burst/` contains HNRS general/network/exclusive catalogs,
native comparator population events, secondary region-aware events,
recording scores, timing/classification diagnostics, and output-processing
sensitivity references. Other folders contain unit, development-grid, null,
seed-isolation, window/bin, shift, duration, and runtime references.

An archived result table is not proof that its full analysis was freshly
rerun for this release. Actual validation coverage is recorded separately
in `VERIFICATION.json` and explained in `REPRODUCIBILITY.md`.

## Exclusions

Dataset generators and parent-input extraction programs, human recordings or
results, clinical notes, EMG, waveforms, personal registries, private source
paths, and repository history inside ZIP artifacts are excluded.

The fixed inputs support exact-input reruns, not new synthetic realizations
or arbitrary simulation-design changes.
