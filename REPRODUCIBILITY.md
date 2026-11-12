# Reproducibility scope

## Tested environment

- Python 3.13.3
- NumPy 2.4.2
- pandas 3.0.1
- SciPy 1.17.1

These versions are recorded in `requirements.txt`. No MATLAB installation or dataset generation is needed to run the included Python HNRS runner on the ready-to-use archived inputs.

## What was checked

1. The HNRS pipeline modules match the archived revised-benchmark implementation. Implementation identifiers are `rs-v3` and `b1-v2` for window shuffling.
2. Configuration input registries are empty, proxy input is disabled, and output defaults use relative paths. Actual benchmark settings come from `methods/profiles/`.
3. A native network run on trial 01 completed with 75 retained units, 476 unit detections, 35 regional detections, and seven native network detections. These are native stage counts, not the final consolidated scoring counts.
4. The burst-pause evaluation trial 001 run completed with 49 detections: TP 49, FP 0, FN 3, F1 0.970297, Jaccard 0.942308. This reproduces its archived HNRS-on reference scores.
5. Rescoring the supplied final network catalogs reproduces the saved HNRS mean recording-level Jaccard values: 0.936590 for general events, 0.799552 for regional classification, and 0.866667 for network classification.
6. Release verification checks every supplied file against `MANIFEST.sha256`. The publication review also inspected the MAT schemas and screened textual files for selected private-path and credential patterns.

These checks are smoke tests and saved-catalog score reproduction, not a rerun of all detectors on every input or a formal confidentiality guarantee.

## Archived network settings

The network profile uses nominal unit/regional/network cutoffs of 0.08/0.05/0.03, 75th-percentile candidate interval limits, 25-s shuffle windows, 2.5-s bins, 10-s maximum source-reference offsets, and seed 123. Network acceptance requires at least two region labels. The archived regional profile permits one distinct unit; it should not be described as requiring two distinct units.

Median contributing burst durations remain the duration-gate basis. The network gate is 20 ms–10 s. The native runner does not add the archived 25-ms final catalog consolidation. Consult the JSON profiles for all settings instead of relying on mutable research-workstation defaults.

## What this draft does not reproduce end to end

- Dataset-generation code is not included for any benchmark. The exact network, burst-pause, and null inputs and their ground truth are supplied instead. This includes the already-extracted single-unit burst-pause inputs.
- The native runner does not rebuild the final consolidated general-event/exclusive-classification catalogs from stage CSVs. `score_benchmarks.py network-burst` evaluates the supplied archived catalogs.
- SPYCODE/logIBEI, the Network-CMA adaptation, and the meaRtools/Otsu implementation/adapters are not included. Their saved synthetic result summaries are not evidence that their runs have been independently reproduced by this package.
- The independent MATLAB original-RS control is not included.
- The synthetic-shift tables are provided, but regeneration of their shifted inputs and complete reruns is not automated here.
- No human analysis is reproduced; neither human input data nor human-derived outputs are supplied.
- The complete research regression-test suite and computational benchmarking are not bundled.

These gaps must remain explicit. The package is a runnable HNRS detector plus archived synthetic inputs and reference analyses, not yet a one-command reproduction of every manuscript result.
