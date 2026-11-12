# Hierarchical Network Rank Surprise (HNRS)

HNRS detects bursting in three linked stages: individual units, predefined recording regions, and networks spanning regions. This repository contains the archived HNRS implementation, benchmark-specific settings, simulated datasets with ground truth, and reference results for the synthetic analyses.

## Repository layout

```text
methods/hnrs/             HNRS source, sanitized configuration, retained GPL notice
methods/profiles/         Exact network, burst-pause, and network-null settings
datasets/network_burst/   60 simulated multiregion recordings and ground truth
datasets/burst_pause/     50 simulated single-unit recordings and ground truth
datasets/network_null/    21 pairs of simulated network-null recordings
results/                 Synthetic reference scores, timing, null counts, shift tables
run_hnrs.py              Run native HNRS stages on a spikeTime MAT input
score_benchmarks.py      Rescore supplied final network catalogs or a unit-stage run
verify_release.py        Verify the exact release file list and checksums
DATA.md                  Dataset provenance, formats, and analysis conventions
REPRODUCIBILITY.md       Tested commands and limits of the included workflows
THIRD_PARTY.md           Attribution and licensing scope
MANIFEST.sha256          Per-file checksums
```

## Included data

All 152 bundled recordings are simulated, not patient recordings:

| Dataset | Recordings | Duration | Contents |
| --- | ---: | ---: | --- |
| Synthetic Regional–Network Burst Benchmark | 60 | 250 s each | 75 units across five groups, 12 regional and six multiregion targets, plus isolated single-unit challenges |
| Burst-Pause Benchmark | 50 | 30 s each | One selected simulated unit per recording with planted burst intervals |
| Network-null controls | 42, organized as 21 pairs | 250 s each | Background only, paired with the same simulated background plus independently timed regional bursts |

The null release includes **all archived source IDs 1–21**, including ID 8. The corresponding table contains 420 analyses: two conditions × 21 recordings × ten internal surrogate seeds. These are repeated analyses of 42 inputs, not 420 independent recordings.

Anatomical-looking names in the MAT files are simulator labels. No human sEEG recordings, patient metadata, human-derived detections or tables, human rasters, or human interactive figures are included. See [DATA.md](DATA.md) for the simulation lineage and exact file meanings.

## Quick start

Tested with Python 3.13.3. From this repository's root:

```sh
python -m pip install -r requirements.txt
python verify_release.py
python run_hnrs.py datasets/network_burst/trials/trial01/spikeTime.mat runs/network-trial01
python run_hnrs.py datasets/burst_pause/evaluation/trial001/spikeTime.mat runs/burst-pause-trial001 --mode burst-pause
```

Use a new or empty output directory. To run a null input:

```sh
python run_hnrs.py datasets/network_null/background_only/trials/trial01/spikeTime.mat runs/null-trial01 --mode network-null --seed 123
```

For the unit-stage ranking-reference ablation, add `--shuffle-off` to a burst-pause run. Outputs include stage detections, effective settings, and the input checksum. No ground truth is read by the detector.

## Scoring

```sh
python score_benchmarks.py burst-pause runs/burst-pause-trial001/hnrs_on_unit_bursts.csv datasets/burst_pause/evaluation/trial001/unit_ground_truth.csv
python score_benchmarks.py network-burst datasets/network_burst
```

The network command rescores the supplied **final HNRS catalogs**. It does not recreate them from a new native run. The archived network analysis used an additional 25-ms HNRS event consolidation and catalog-construction step. The quick-start runner outputs native stages without that final construction. Dataset generators and comparator source/adapters are not bundled; the actual archived datasets are provided ready to use. These boundaries are detailed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Credits and licensing

HNRS builds on original Rank Surprise: [Gourévitch and Eggermont (2007)](https://doi.org/10.1016/j.jneumeth.2006.09.024). The detector retains the original credit to [Subhasis Ray's MATLAB-to-Python port](https://gist.github.com/subhacom/4336215) and its GPL-3.0-or-later notice. The license text is in `methods/hnrs/COPYING`. See [THIRD_PARTY.md](THIRD_PARTY.md).

No separate reuse license is asserted for the bundled datasets. Public redistribution and dataset-license terms require approval from their rights holder before this draft is published.
