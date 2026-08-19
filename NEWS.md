GilbertCurves.jl release notes
==============================

main
----

v0.1.1
------

- Fixed `gilbertindices` and `GilbertCurves.gilbertorder` to return an empty
  list for empty domains (e.g. `(0, 5)`), instead of recursing without bound.
