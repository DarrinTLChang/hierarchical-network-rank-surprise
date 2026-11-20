# Fixed-setting comparator reproduction

This directory contains the study adapters for **logIBEI 100 ms**,
**Network-CMA k = 0.75**, and **meaRtools Sigma 45**, the general-selected
settings used throughout the manuscript. No synthetic generators are included.

## Dependencies

Fresh detection requires licensed MATLAB. Both archived and fresh parity runs
used **R2025b Update 4, 25.2.0.3150157, PCWIN64**, one numerical thread.
SPYCODE uses smooth() from Curve Fitting Toolbox and legacy uddpvparse/
dspopts.findpeaks helpers from Signal Processing Toolbox. Network-CMA uses
skewness() from Statistics and Machine Learning Toolbox. The included meaRtools
MATLAB port does not require R. MATLAB and its toolboxes are not supplied.

Git and internet access are needed once for the pinned external source trees:

    python comparators/setup_external.py
    python comparators/setup_external.py --verify-only

The default directories are external_dependencies/SpyCode and
external_dependencies/NetworkWideAdaptiveBurstDetection at repository root.
Downloaded upstream trees are not bundled in the release. See
[THIRD_PARTY.md](../THIRD_PARTY.md), including the file-specific MathWorks
notices in upstream SPYCODE. Python requirements suffice to score saved catalogs
without MATLAB.

## Fresh detection

In MATLAB, from the repository root:

    addpath('comparators/matlab');
    run_comparators('datasets/network_burst', 'runs/comparators');

The input can also name a trials/ directory. The optional third argument selects
recording IDs, e.g. [1 15 30 45 60]. The optional fourth argument names a directory
containing the pinned external checkouts. A separate development cohort must be
supplied by its explicit archived-input path. No inputs are generated or replaced.
Use a **new** output directory.

Common unit eligibility is SNR 1.2–25, whole-recording firing rate strictly
greater than 0.8 Hz, and at least two spikes. All methods receive the same
retained units. meaRtools recombines clusters by physical-electrode name and
retains its native crop/filter semantics. Sigma 45 is a sample-based parameter,
not a 45-ms standard deviation. No common merge, extra duration filter, or new
per-region participation threshold is imposed on comparator population events.
The driver checks cohort metadata existence but reads no planted-label values.

## Output contract

Each trialNN/ directory contains:

- native_population_events.csv: method, trial, master_seed, event_id,
  prediction_id, start_ms, end_ms, kind, regions. kind=population and empty
  regions intentionally preserve unlabeled native population calls.
- support_events.csv: method, trial, master_seed, event_id, region, start_ms,
  end_ms, unit_id. LogIBEI support uses the exact accepted native onset-group
  members, not all overlapping unit bursts. CMA support uses unit bursts.
  meaRtools support uses region-local detections at the same Sigma 45.
- recording_settings.json: fixed settings, zero-inclusive counts, exposures,
  retained-unit number, and actual MATLAB version.
- Native per-unit, threshold, population, and signal tables for traceability.

The comparator master_seed=0 is a deterministic marker, not an HNRS surrogate
seed. Support intervals retain their full original endpoints; secondary
region-aware classification clips overlap inside the native parent interval and
requires positive concurrent support from distinct regions. It attaches labels
without changing event boundaries. Primary native-output network-target scoring
does not apply anatomical labels. These evaluations are separate in the scorer.

## Checks

    python -m unittest discover -s comparators -p "test_*.py"
    python comparators/verify_run.py runs/comparators path/to/reference_catalog.csv

The verifier checks fresh population boundaries and, if supplied in the
reference, secondary classes derived from native support. A multi-task reference
can be filtered using --reference-task general_events (the default). Boundary
tolerance is 1e-7 ms. Rescoring saved references is not a fresh detector rerun.
