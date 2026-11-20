# Hierarchical Network Rank Surprise (HNRS)

**Manuscript reproducibility package, version 2026.11.19.** This repository
contains HNRS code, fixed synthetic inputs and ground truth, comparator
adapters, analysis settings, scoring workflows, and current reference results.
Synthetic dataset generators are deliberately excluded. The detector's internal
surrogate-reference procedures remain included as part of HNRS.

## Package contents

- `methods/hnrs/`: frozen HNRS engine with retained GPL notices.
- `methods/profiles/`: current network, unit-only, and null-control profiles.
- `datasets/recording_manifest.csv`: portable input paths, identities, and hashes.
- `datasets/`: 50 Burst-Pause, 30 development, 60 multiregion evaluation,
  42 paired-null inputs, 120 fixed shifted inputs, and one synthetic illustration.
- `comparators/`: MATLAB adapters and pinned upstream dependency setup.
- `results/`: synthetic reference catalogs, scores, and sensitivity tables.
- `reproduce.py`: batch HNRS execution, fresh-output scoring, and reference rescoring.
- `score_benchmarks.py`: matching, anatomical attribution, and seed-aware summaries.
- `DATA.md`, `REPRODUCIBILITY.md`, `THIRD_PARTY.md`: formats, tested scope, attribution.
- `MANIFEST.sha256`, `verify_release.py`: exact release-file verification.

All 303 MAT inputs are simulated. No human recordings, patient metadata,
human-derived catalogs or figures, dataset generators, or private workstation
files are included. Anatomical-looking channel strings are simulator labels.

## Quick start

Tested with Python 3.13.3 and the versions in `requirements.txt`. Run commands
from the repository root and choose new or empty output directories.

```sh
python -m pip install -r requirements.txt
python verify_release.py
python reproduce.py run-hnrs --cohort network_burst --trials 1 --seeds 123 --out runs/hnrs-smoke
python reproduce.py score-runs --runs runs/hnrs-smoke --out runs/hnrs-smoke-scores
```

This runs the detector, constructs the standard reported catalogs, and scores
those **fresh outputs** against unpadded truth. Native stage inputs remain
unchanged; reported HNRS catalogs use standard 25-ms consolidation.

## Main HNRS evaluation

```sh
python reproduce.py run-hnrs --cohort network_burst --jobs 4 --out runs/hnrs-network
python reproduce.py score-runs --runs runs/hnrs-network --out runs/hnrs-network-scores
```

Default network seeds are 123, 456, and 789, with two separately derived
reference streams for each recording. Expected mean event Jaccards are
**1.000** for general detection, **0.934** for secondary regional-only
classification, and **0.940** for network targets. Scores average seeds
within recording; three seeds are not three independent recordings.

## Unit-only comparison

```sh
python reproduce.py run-hnrs --cohort burst_pause --jobs 4 --out runs/hnrs-unit
python reproduce.py score-runs --runs runs/hnrs-unit --out runs/hnrs-unit-scores
python reproduce.py run-hnrs --cohort burst_pause --shuffle-off --jobs 4 --out runs/rs-unit
python reproduce.py score-runs --runs runs/rs-unit --out runs/rs-unit-scores
```

Expected mean F1/Jaccard are approximately **0.892/0.824** for HNRS and
**0.348/0.215** for original RS. The unit-only protocol disables duration
gates. RS retains native intervals; HNRS consolidation changes none of
these 50 benchmark inputs' reported intervals.

## Rescore all methods without MATLAB

```sh
python reproduce.py score-reference --out runs/reference-scores
```

This recomputes all 16 current method/task summaries from supplied catalogs.
It **does not rerun** the detectors.

## Rerun comparators

The adapters use logIBEI 100 ms, Network-CMA `k=0.75`, and meaRtools Sigma 45.
Primary comparator population outputs retain their native acceptance and
reporting rules; no HNRS consolidation/duration rules are silently added.

```sh
python comparators/setup_external.py
python comparators/setup_external.py --verify-only
```

Then run MATLAB from this repository's root:

```matlab
addpath('comparators/matlab');
run_comparators('datasets/network_burst', 'runs/comparators');
```

Then score those fresh adapter outputs:

```sh
python reproduce.py score-comparators --runs runs/comparators --out runs/comparator-scores
```

Comparator reruns require MATLAB and Signal Processing, Statistics and Machine
Learning, and Curve Fitting Toolboxes. The tested release is R2025b Update 4.
Upstream dependencies stay in gitignored `external_dependencies/`; they are
not redistributed in this package. See `comparators/README.md` and
`THIRD_PARTY.md`.

## Optional summary plots

```sh
python -m pip install -r requirements-figures.txt
python plot_results.py --scores results/network_burst/recording_scores.csv --out runs/plots
```

This makes a convenience PNG/SVG summary from the selected score table and
exports its plotted means. It is not a new detector run or a reproduction of
every manuscript figure's layout.

## Interpretation

General recovery and network-target selectivity are different objectives.
In the primary network-target comparison, a native population event matching
a true single-region burst is a non-target call, not necessarily spurious
activity. The secondary region-aware comparison adds concurrent two-region
attribution without adding an HNRS-style three-onset gate to comparators.

Event Jaccard is `TP/(TP+FP+FN)`, not temporal IoU. Rank Surprise candidate
scores and null-call rates do not establish recording-wide error control or
human false-positive rates. Evaluation inputs had informed HNRS development;
they are not independent external validation.

The package supports exact-input reruns, not regeneration of synthetic data
or reproduction of restricted human analyses. See `REPRODUCIBILITY.md`.

## Citation and licensing

Version information is in `VERSION` and `CITATION.cff`. HNRS builds on
[Gourevitch and Eggermont (2007)](https://doi.org/10.1016/j.jneumeth.2006.09.024).
The detector retains Subhasis Ray's Python-port credit and GPL-3.0-or-later
notice. Applicable license texts and comparator source attribution are in
`THIRD_PARTY.md`.

No new blanket license is assigned to the datasets or every repository file.
Contact the corresponding author about dataset reuse permissions.
