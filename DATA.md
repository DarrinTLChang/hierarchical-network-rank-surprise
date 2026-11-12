# Datasets and provenance

All bundled spike times are from simulations. None of these inputs is a surrogate made from a human recording. Simulator electrode labels resemble anatomical names for compatibility with the HNRS input parser; they do not identify patient recording sites.

## Shared format

Each `spikeTime.mat` contains one MATLAB variable, `spikeTime`. Channel fields are `electrode`, `time`, `snr`, and `dataSegmentLength`. Spike times are in milliseconds; `dataSegmentLength` is in seconds. Empty cluster cells can be present.

Network-stage unit labels use zero-based cluster indices. The burst-pause runner instead exports one-based `Cluster` values to match its selected-unit ground truth. These conventions should not be interchanged.

## 1. Synthetic Regional–Network Burst Benchmark

Directory: `datasets/network_burst/`

- 60 simulated 250-s recordings, each with 75 units across five groups of sizes 10/10/15/20/20.
- Each recording has 12 regional-only and six multiregion population targets. Regional-only recruitment is drawn from 25%–90% of the available units; network recruitment follows the archived generator settings.
- The underlying design uses synthetic Gaussian unit-onset jitter, signed regional anchor offsets from −18 to +18 ms, and independent smooth regional background-rate envelopes with ±50% modulation. The programmed background baseline is at least 1 Hz; this is not a guarantee that every instantaneous or sampled firing rate exceeds 1 Hz.
- This release is the **isolated-single-unit-overlap challenge** variant, not just the earlier baseline. Across the 60 recordings, 5,583 simulated spikes were added to six regional targets per recording: two brief remote-unit overlaps, two substantial overlaps, and two remote bursts separated by a short gap. The 18 population targets and their labels were retained. Additions have separate annotations.
- Backgrounds and burst spike clusters were generated using seeded random simulation. Donor templates for the isolated-unit additions are other simulated unit bursts, not patient spikes. The inspected generation chain does not load human spike times or fit parameters to a human recording.

Each `trials/trialNN/` contains exactly:

| File | Meaning |
| --- | --- |
| `spikeTime.mat` | Final synthetic detector input |
| `channel_map.csv` | Simulator electrode/cluster-to-group mapping |
| `event_manifest.csv` | Planted population targets |
| `event_components.csv` | Region-level components of those targets |
| `unit_event_truth.csv` | Planted unit-level activity |

Use `actual_start_ms` and `actual_end_ms` for the unpadded population target intervals. Historical padded `score_*` columns remain in the source metadata but are not the revised scoring boundaries. Some event-level count/rate descriptors describe the pre-addition simulation; use `added_spikes.csv`, `challenge_manifest.csv`, and `recording_audit.csv` when examining the final input spike counts.

`reference_catalogs/general_events.csv` and `exclusive_events.csv` contain the archived final HNRS catalogs. The first is scored without anatomical class; the second contains regional-only/network assignments. Matching is one-to-one positive temporal overlap against unpadded targets, maximizing the number of matches before summed IoU.

`results/network_burst/` contains reference recording scores, summaries, and temporal-error summaries. Comparator summary values are saved results, not newly reproduced comparator runs.

## 2. Burst-Pause Benchmark

Directory: `datasets/burst_pause/`

- 50 simulated 30-s single-unit inputs: 20 in `calibration/`, 30 in `evaluation/`.
- 33,478 spikes and 2,811 planted bursts across these selected-unit inputs.
- Each input is the first fixed unit/electrode selected from a larger simulated parent recording, not selected by detection performance.
- All spikes in this benchmark belong to planted bursts. It is a controlled burst-pause test, not a tonic/rare-burst background test.
- Parent burst schedules, spike counts, and gamma/lognormal/Weibull interval patterns were generated with seeded random streams. SNR values are synthetic.

Each trial has `spikeTime.mat` and `unit_ground_truth.csv`. The manifest uses relative paths and input checksums. Unit scoring uses chronological one-to-one positive-overlap matching, maximizing match count before summed overlap duration.

The exact selected-unit inputs and their ground truth are supplied ready to use; no parent-recording generator or selected-unit extraction code is included. `results/burst_pause/` contains per-recording and summary reference metrics, including the original RS control.

## 3. Network-null controls

Directory: `datasets/network_null/`

This release contains **all 21 archived pairs, source IDs 1–21**, as selected for release review. It is not the separate 20-pair subset that omitted source ID 8.

- `background_only/`: 21 × 250-s simulated backgrounds, without planted bursts.
- `independent_regional/`: the paired simulated backgrounds with 32 intact synthetic regional components independently repositioned within 10–240 s. Within-region overlaps are disallowed by this generator; coincidences between different regions are allowed. There are no deliberately planted network bursts.
- Both conditions use 75 simulated units, the same data format, and simulator channel labels.

Each background trial contains `spikeTime.mat` and `channel_map.csv`. Each independent-regional trial additionally contains `regional_components.csv` and `network_ground_truth.csv`.

`results/network_null/network_counts_21pairs_10seeds.csv` has 420 rows: two conditions × 21 inputs × ten surrogate seeds (123, 456, 789, 1001–1007). Counts use the archived 25-ms HNRS consolidation. Those repeated surrogate runs do not add independent input recordings.

## 4. Synthetic-shift reference tables

`results/synthetic_shift/` contains saved offsets, paired detection counts, and a summary from the deliberately constrained synthetic desynchronization control. Shifted MAT copies are not included. The offsets were selected to separate planted regional components; these tables describe that stress test, not an unconstrained false-positive calibration or a human empirical null.

## Exclusions and approval

Dataset-generation code, human recordings, sorted human spike trains, human-derived detection catalogs/tables, human figures and HTML, patient metadata, manuscript/reviewer files, personal input registries, logs, private source paths, and repository history are not part of this release.

Structural checks and simulation provenance support the proposed synthetic-only scope. They do not replace rights-holder or institutional permission to redistribute the data. Dataset permission and reuse-license terms must be confirmed before publication.
