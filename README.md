# GilbertCurves.jl

Space-filling curves for rectangular domains of arbitrary size.

`GilbertCurves.jl` is a Julia implementation of the generalized Hilbert
("gilbert") space-filling curve algorithm by
[Jakub Červený](https://github.com/jakubcerveny/gilbert). It extends the
classic Hilbert curve beyond power-of-two square domains to arbitrary
rectangles. Currently only 2D domains are supported.

|||
|------------:|:-------------------------------------------|
| **Version** | [![version][version-img]][version-url]     |
| **License** | [![license][license-img]][license-url]     |
| **Tests**   | [![gha ci][gha-ci-img]][gha-ci-url]        |

[version-img]: https://juliahub.com/docs/General/GilbertCurves/stable/version.svg
[version-url]: https://juliahub.com/ui/Packages/General/GilbertCurves
[license-img]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: LICENSE
[gha-ci-img]: https://github.com/CliMA/GilbertCurves.jl/actions/workflows/CI.yml/badge.svg?branch=main
[gha-ci-url]: https://github.com/CliMA/GilbertCurves.jl/actions/workflows/CI.yml?query=branch%3Amain

## Features

- Generalized Hilbert curves for any m × n rectangular domain, with no
  power-of-two constraint.
- `gilbertindices` returns the curve as a vector of `CartesianIndex`;
  `GilbertCurves.gilbertorder` returns the elements of a matrix in curve order;
  `GilbertCurves.linearindices` returns the inverse map (the curve position of
  each cell).
- The traversal direction is selectable via the `majdim` keyword.

## Installation

```julia
import Pkg; Pkg.add("GilbertCurves")
```

## Quick Example

```julia
using GilbertCurves
using Plots

# Generate the space-filling curve for a 67x29 grid
list = gilbertindices((67, 29))

# Plot the continuous path
plot([c[1] for c in list], [c[2] for c in list], line_z=1:length(list), legend=false)
```

![Gilbert curve on 67 x 29 elements](https://raw.githubusercontent.com/CliMA/GilbertCurves.jl/main/img/67x29.png)

## Documentation

Documentation is contained in this README and in the docstrings (e.g.
`?gilbertindices` in the REPL). The continuity properties of the curves are
detailed below.

## Integration with CliMA models

[ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl) uses this package to
order the elements of its 2D and cubed-sphere grids along a space-filling
curve, so that partitioning the curve into contiguous segments yields
spatially localized subdomains for distributed runs.

## Contributing

Please see the [CliMA DeveloperGuides](https://github.com/CliMA/DeveloperGuides)
for style, documentation, and contribution conventions.

## Continuity of the curves

The curve visits every cell exactly once, moving between edge-adjacent cells,
with one exception: **when the larger dimension of the domain is odd and the
smaller dimension is even, the curve contains a single diagonal step.** For
example, the 15 × 12 curve below has one diagonal step near its far corner:

```julia
julia> list = gilbertindices((15,12));

julia> plot([c[1] for c in list], [c[2] for c in list], line_z=1:length(list), legend=false)
```

![Gilbert curve on 15 x 12 elements](https://raw.githubusercontent.com/CliMA/GilbertCurves.jl/main/img/15x12.png)

### Why: a parity argument

No algorithm can avoid this diagonal step while preserving the corner-to-corner
endpoints. Color the m × n grid like a checkerboard and take m odd, n even,
m > n (the odd-by-even case with `majdim = 1`):

1. Each orthogonal step moves to a cell of the other color, so a path visiting
   all m·n cells makes m·n − 1 steps and ends on the same color it started on
   only if m·n is odd. With n even, m·n is even, so the endpoints of any
   continuous path must have different colors.
2. The curve is required to run from `(1,1)` to `(m,1)`. With m odd, these two
   corners have the same color.

The two requirements contradict each other, so no continuous corner-to-corner
path exists: the curve must either contain one diagonal step (which stays on
the same color, breaking the alternation) or give up the corner endpoint. This
implementation takes the diagonal step, except when the smaller dimension is 2,
where it instead stays continuous and ends away from the corner.

### Continuous configurations

Every other rectangle yields a fully continuous curve. In particular:

- **Square grids** (n × n): if n is odd, the cell count is odd and same-colored
  endpoints are consistent; if n is even, the corner endpoints have different
  colors. Either way the parity obstruction vanishes. Cubed-sphere faces are
  square, so their curves are always continuous.
- **Even-by-anything grids**: whenever the larger dimension is even, the
  corner endpoints have different colors and the curve is continuous.

The test suite verifies these continuity properties for all domain sizes up to
20 × 20.
