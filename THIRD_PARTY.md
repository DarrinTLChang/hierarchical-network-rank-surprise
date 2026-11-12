# Attribution and licensing scope

## Original Rank Surprise and Python-port credit

The statistical method builds on:

Gourévitch, B. and Eggermont, J. J. (2007). A nonparametric approach for detection of bursts in spike trains. *Journal of Neuroscience Methods*. DOI: https://doi.org/10.1016/j.jneumeth.2006.09.024

`methods/hnrs/pipeline/detection.py` retains its existing Subhasis Ray MATLAB-to-Python-port attribution and GPL-3.0-or-later notice. The GPL version 3 text is preserved in `methods/hnrs/COPYING`. This draft does not remove or replace those notices.

## Comparator references

Comparator implementations/adapters are **not bundled**. Saved comparator scores are included among the synthetic reference results.

- SPYCODE: https://github.com/barbaLab/SpyCode; archived source commit `fb9bc345d562939a4c6f4c90002e5f689de45db1`.
- NetworkWideAdaptiveBurstDetection: https://github.com/kerstinlenk/NetworkWideAdaptiveBurstDetection; archived source commit `9de22b0c2dde0dce29adfa207245e15d1f3805a4`. The study's Network-CMA population-threshold adaptation must be distinguished from that repository's original formulation.
- meaRtools/Otsu: the archived comparisons used a MATLAB implementation of that approach, not a bundled copy of meaRtools.

## Outstanding license decisions

This draft preserves existing third-party notices. It does not assign a new blanket license to every original project file or to the datasets. Before publication, the rights holder must approve redistribution and specify the license scope for original HNRS additions and for the simulated datasets/results. No dataset-use rights should be inferred merely from the fact that a GitHub repository is publicly visible.
