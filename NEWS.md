GilbertCurves.jl release notes
==============================

main
----

v0.1.1
------

- Fixed `gilbertindices` and `GilbertCurves.gilbertorder` to return an empty
  list for empty domains (e.g. `(0, 5)`), instead of recursing without bound.
- Raised the minimum supported Julia version to 1.10 (LTS); CI now tests 1.10,
  the current stable release, and nightly.
