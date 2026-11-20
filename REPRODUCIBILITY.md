# Reproduction workflow and tested scope

This release reruns detectors on fixed synthetic inputs. It does not include
dataset generators or human recordings.

## Environment

- Python 3.13.3, NumPy 2.4.2, pandas 3.0.1, SciPy 1.17.1.
- Optional plots: Matplotlib 3.10.8 (`requirements-figures.txt`).
- Comparators: MATLAB R2025b Update 4 with Signal Processing, Statistics and
  Machine Learning, and Curve Fitting Toolboxes, plus pinned upstream sources.

Use new or empty output directories. Scripts do not delete old analyses.
Parallel HNRS execution uses separate processes because detector configuration
is module-level; do not parallelize it in threads.

## Settings and reporting

Network/development profiles use unit/regional/network cutoffs 1%/3%/15%,
75th-percentile candidate limits, a minimum of three events, 25-second shuffle
windows, 2.5-second bins, and a 10-second internal offset bound. Regional
acceptance permits repeated onsets from one unit; network acceptance requires
two recording regions. No new three-distinct-unit or maximum onset-spread
criterion is imposed.

Native stage inputs are unchanged. Standard 25-ms consolidation constructs
reported catalogs separately. General detection uses pooled regional outputs.
Exclusive classification retains network events and regional events not sharing
a native contributing unit burst with an accepted network event. Intrinsic
duration gates precede consolidation; regional/network duration reporting uses
contributing unit durations.

The unit-only protocol uses a 1% cutoff, 3-second windows, 0.3-second bins, base
seed 123, and no duration gate. RS outputs remain native. HNRS consolidation
does not change any reported intervals in the 50-input unit benchmark.

## Seed contract

Primary network/development analyses use master seeds 123/456/789. Each stream
seed is the first four big-endian bytes of SHA-256 over compact JSON
`["hnrs-joint-alpha-v1", master_seed, "network_burst", recording_id, stream]`.
Stream names are `unit-shuffle` and `source-offset`. Recording IDs distinguish
development from evaluation; they are not the display trial number.

Null analyses use literal paired reference seeds 123, 456, 789, and 1001-1007.
The unit protocol retains its literal base seed and internal unit/bounds hash.
Window/bin sensitivity uses literal unit seeds 123/456/789 and source-offset
seed 123, so its default is not the same random-reference realization as the
primary benchmark.

Average repeated seeds within each recording before recording means and
bootstrap intervals. Precision is also reported pooled over calls. Repeated
seeds do not increase the independent sample count.

## Commands

Main evaluation and unit-only commands are in `README.md`. One explicit run:

```sh
python run_hnrs.py datasets/network_burst/trials/trial01/spikeTime.mat runs/direct-trial01 --recording-id 73001 --master-seed 123 --trial 1
```

Additional fixed-input analyses:

```sh
python reproduce.py run-hnrs --cohort network_development --trials 1 --out runs/development-smoke
python reproduce.py run-hnrs --cohort network_null --trials 1 --seeds 123 --out runs/null-pair-smoke
python reproduce.py score-runs --runs runs/null-pair-smoke --out runs/null-pair-counts
python reproduce.py run-hnrs --cohort network_shift --trials 1 --seeds 123 --out runs/shift-smoke
python reproduce.py score-runs --runs runs/shift-smoke --out runs/shift-counts
```

Without `--trials`, all cohort inputs are used. Default seeds are three for
primary network/development/shift, one for unit-only, and ten for null analyses.
`--condition` selects a named manifest condition. HNRS shift counts require
positive overlap with 5-245 seconds; null counts use the full interval.

Comparator setup/driver commands are in `comparators/README.md`. Fresh outputs
include native population events, detected regional support, and settings.
`score_benchmarks.assign_region_labels` supplies concurrent two-region
attribution without changing population-event boundaries.

## Matching and objectives

Population matching requires positive overlap against unpadded truth,
maximizes match count, then summed temporal IoU. Primary network-target scoring
uses native comparator population calls, including real single-region bursts
as non-target calls. Secondary anatomical matching is network-first, followed
by regional predictions against remaining targets. Wrong-class assignments
are FP for the predicted class and FN for the true class; regional scoring
assesses class, not exact region identity.

Unit matching instead uses chronological one-to-one matching, maximizing
match count then summed overlap duration. It does not use IoU as its tie-break.

Primary comparator outputs do not receive HNRS consolidation or added output
duration filters. Supplied output-processing sensitivity tables describe a
separate comparison, not the primary native-output scores.

## Parameter selection

The 30 development inputs use the complete protocol including isolated-unit
challenges. HNRS explored 216 cutoff combinations; each comparator explored
13 values of one parameter. Tables and selection metadata are supplied in
`results/parameter_selection/`.

The reported 1%/3%/15% prioritizes general recovery, retaining the network
cutoff through the documented tie-break with a conditional network sweep as
supporting evidence. It is not the highest network score across the full grid.
Search budgets differ, and evaluation recordings informed HNRS development.
The evaluation is not fully blinded or independent external validation.

## Tested coverage

`VERIFICATION.json` records actual release checks. Fresh native HNRS and
unit-control outputs are compared with current archives; reference rescoring
is checked separately from detector execution. Comparator verification names
the trials/runtime actually tested rather than treating saved catalogs as
fresh reruns. Tests exercise seeds, matching, and reporting contracts.

Reference tables also cover parameter grids, null seeds, seed-isolation,
window/bin changes, timing/duration reporting, fixed shifts, and runtime.
Not every sensitivity grid is freshly rerun for this release. Runtime figures
are workload-specific and not cross-platform guarantees. No human analysis
is reproduced.

## Deliberate limitations

Dataset generators are excluded: this is exact-input reproduction, not
synthetic-data regeneration. Comparator execution requires the specified
MATLAB environment and upstream dependencies. Rank Surprise cutoffs and null
rates are not calibrated recording-wide significance levels or estimates of
human false-positive rates.
