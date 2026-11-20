# Release 2026.11.19

- Updated the default reported HNRS profile to 1%/3%/15%, with the current
  recording-derived reference streams and standard reported-catalog construction.
- Added end-to-end fresh-output scoring, seed-aware primary and secondary tasks,
  batch commands, and original-RS-equivalent unit-only control execution.
- Added portable comparator adapters at logIBEI 100 ms, Network-CMA 0.75,
  and meaRtools Sigma 45, with pinned external dependency setup.
- Added development, fixed-shift, and illustration inputs plus additional truth.
  The 152 inputs already present in the repository remain byte-identical.
- Updated reference outputs and documented actual rerun coverage in
  `VERIFICATION.json` and `comparators/verification.json`.
- Added a portable manifest, data dictionary, plotting helper, citation metadata,
  and byte-preserving Git attributes for release checksum consistency.
- Explicitly excluded synthetic dataset generators and all human inputs/results.

Primary network execution now requires the recording identity to derive the
reference seeds. Use `reproduce.py` with the input manifest, or explicitly pass
`--recording-id` to `run_hnrs.py`. Read the current README rather than using
commands or profiles copied from another package version.
