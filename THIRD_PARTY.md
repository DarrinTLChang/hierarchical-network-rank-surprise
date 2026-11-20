# Attribution and licensing scope

## Original Rank Surprise and Python-port credit

The statistical method builds on:

Gourévitch, B. and Eggermont, J. J. (2007). A nonparametric approach for detection of bursts in spike trains. *Journal of Neuroscience Methods*. DOI: https://doi.org/10.1016/j.jneumeth.2006.09.024

`methods/hnrs/pipeline/detection.py` retains its existing Subhasis Ray MATLAB-to-Python-port attribution and GPL-3.0-or-later notice. The GPL version 3 text is preserved in `methods/hnrs/COPYING`. This draft does not remove or replace those notices.

## Comparator adapters and external dependencies

The portable study adapters are included in comparators/matlab/. The external
SPYCODE and NetworkWideAdaptiveBurstDetection source trees are not bundled.
The optional comparators/setup_external.py script downloads pinned checkouts
into external_dependencies/, which is excluded from the release file set.
Source availability alone is not treated as permission to redistribute code.

### SPYCODE / logIBEI

- Upstream: https://github.com/barbaLab/SpyCode
- Pinned commit: fb9bc345d562939a4c6f4c90002e5f689de45db1
- Upstream license file: LICENSE.md, GNU Lesser General Public License version 3.
- The study adapter invokes the original detector with a 100-ms histogram
  peak-search limit, 0.90 network void parameter, and native 20% participation
  relative to burst-active units. It does not replace the adaptive threshold
  or impose regional participation during native detection.
- File-specific notices matter: upstream findpeaksISI.m and findpeaksIBEI.m
  carry MathWorks copyright notices and use proprietary MATLAB toolbox helpers.
  The top-level LGPL file is not assumed to override those notices.
  Those upstream files are therefore not redistributed in this source package.
  Users obtain the pinned upstream checkout separately and need appropriately
  licensed MATLAB/toolboxes. This is not a legal determination about every
  third-party file in the upstream repository.

### Network-CMA

- Upstream: https://github.com/kerstinlenk/NetworkWideAdaptiveBurstDetection
- Pinned commit: 9de22b0c2dde0dce29adfa207245e15d1f3805a4
- Upstream README explicitly states GPL version 3 or later; LICENSE supplies
  the GPL version 3 text.
- run_valkki_net.m calls the unchanged pooled-ISI, per-unit network-wide CMA
  detector. The separate population count-signal threshold at mean plus
  0.75 population standard deviations is the study adaptation in
  run_comparators.m, not a native population-event rule from that repository.

### meaRtools/Otsu MATLAB port

- Upstream: https://github.com/igm-team/meaRtools
- Audited commit: 0b04c9c8421a8bc23fbe15da59f8b9a092e6a139; package 1.0.4.
- Source path: meaRtools/R/NetworkBurst_functions.R.
- Source SHA-256:
  5544d352f04036ea196a05c20c1e76b63b28ef5212a6de5cc691e78aa1e05633.
- Upstream licensing is GPL version 3 or later. The source-faithful MATLAB
  port and its filter/threshold helpers retain this attribution and licensing
  context; the GPL version 3 text is also present in methods/hnrs/COPYING.
- The included port does not call R at runtime. The fixed profile uses Sigma
  45, four minimum active electrodes for processing, no additional event-level
  regional participation threshold, native cropping, and source time indexing.

### References and runtime requirements

SPYCODE: Bologna et al. (2010), Investigating neuronal activity by SPYCODE
multi-channel data analyzer. Network-CMA: Valkki et al. (2017),
https://doi.org/10.3389/fncom.2017.00040; underlying CMA: Kapucu et al. (2012),
https://doi.org/10.3389/fncom.2012.00038. meaRtools: Gelfman et al. (2018),
https://doi.org/10.1371/journal.pcbi.1006506.

MATLAB and its toolboxes are not supplied or relicensed by this repository.
Fresh comparator runs were checked using MATLAB R2025b Update 4,
25.2.0.3150157, with Curve Fitting, Signal Processing, and Statistics and
Machine Learning toolbox functions. See comparators/README.md for commands.

## License scope

Existing third-party notices are preserved. No new blanket license is asserted
for every original project file or for the datasets/results. The upstream
license of one component does not automatically grant reuse rights for other
components or data. Rights-holder approval and explicit scope are required for
any additional licensing declaration; public visibility alone grants no such
rights. No synthetic generators or human recording files are included.
