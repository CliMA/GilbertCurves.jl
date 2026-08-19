"""
    GilbertCurves

Generalized Hilbert ("Gilbert") space-filling curves for rectangular domains of
arbitrary size, based on the algorithm by Jakub Červený
(https://github.com/jakubcerveny/gilbert).
"""
module GilbertCurves

export gilbertindices

"""
    gilbertindices(dims::Tuple{Int,Int}; majdim = dims[1] >= dims[2] ? 1 : 2)

Construct a vector of `CartesianIndex` objects ordered by a generalized Hilbert
space-filling curve.

The sequence starts at `CartesianIndex(1, 1)` and ends at
`CartesianIndex(dims[1], 1)` if `majdim == 1`, or `CartesianIndex(1, dims[2])`
if `majdim == 2` (or the closest feasible point).

# Arguments
- `dims`: dimensions of the grid.

# Keyword Arguments
- `majdim = dims[1] >= dims[2] ? 1 : 2`: the major dimension of the traversal
  (1 or 2); defaults to the larger dimension.

# Examples
```julia
julia> gilbertindices((2,2))
4-element Vector{CartesianIndex{2}}:
 CartesianIndex(1, 1)
 CartesianIndex(1, 2)
 CartesianIndex(2, 2)
 CartesianIndex(2, 1)
```

See also [`gilbertorder`](@ref), [`linearindices`](@ref).
"""
gilbertindices(dims::Tuple{Int,Int}; kwargs...) =
    gilbertorder(CartesianIndices(dims); kwargs...)


"""
    gilbertorder(mat::AbstractMatrix; majdim = size(mat,1) >= size(mat,2) ? 1 : 2)

Construct a vector of the elements of `mat`, ordered by a generalized Hilbert
space-filling curve.

The list starts at `mat[1,1]`, and ends at `mat[end,1]` if `majdim == 1` or
`mat[1,end]` if `majdim == 2` (or the closest feasible point).

# Arguments
- `mat`: the matrix to traverse.

# Keyword Arguments
- `majdim = size(mat,1) >= size(mat,2) ? 1 : 2`: the major dimension of the
  traversal (1 or 2); defaults to the larger dimension.

See also [`gilbertindices`](@ref).
"""
function gilbertorder(mat::AbstractMatrix{T}; majdim=size(mat,1) >= size(mat,2) ? 1 : 2) where {T}
    list = sizehint!(T[], length(mat))
    # the recursion in append_gilbert! has no base case for empty blocks
    if isempty(mat)
        return list
    end
    if majdim == 1
        append_gilbert!(list, mat)
    else
        append_gilbert!(list, permutedims(mat,(2,1)))
    end
    return list
end

"""
    append_gilbert!(list, mat::AbstractMatrix)

Recursively append the elements of `mat` to `list` in generalized Hilbert curve
order, traversing along the first dimension of `mat`.

The matrix is split into two or three blocks, transposed as needed so that each
block is again traversed along its first dimension. When the larger dimension
is odd and the smaller even, checkerboard parity makes the required corner
endpoint unreachable by an orthogonal path, and the curve contains one diagonal
step (see the README).
"""
function append_gilbert!(list, mat::AbstractMatrix)
    # 1 |*    |
    #   | )   |
    # a |v    |
    a,b = size(mat)
    if a == 1 || b == 1
        # single in one dimension
        append!(list, mat)
    elseif 2a > 3b
        # long case: split into two
        #   +-----+
        # 1 |*    |
        #   ||    |
        # a2|v    |
        #   +-----+
        #   |*    |
        #   ||    |
        # a |v    |
        #   +-----+
        a2 = div(a,2)
        if isodd(a2) && a > 2
            a2 += 1
        end
        append_gilbert!(list, mat[1:a2,:])
        append_gilbert!(list, mat[a2+1:a,:])
    else
        # standard case: split into three
        #      b2
        #   +---+---+
        # 1 |*->|*   |
        #   |   ||   |
        # a2|   ||   |
        #   +---+|   |
        #   |   ||   |
        # a |<-*|v   |
        #   +---+----+
        a2 = div(a,2)
        b2 = div(b,2)
        if isodd(b2) && b > 2
            b2 += 1
        end
        append_gilbert!(list, permutedims(mat[1:a2,1:b2],(2,1)))
        append_gilbert!(list, mat[:,b2+1:b])
        append_gilbert!(list, permutedims(mat[a:-1:a2+1,b2:-1:1],(2,1)))
    end
end

"""
    linearindices(list::Vector{CartesianIndex{2}})

Construct an integer matrix `M` containing the integers `1:length(list)` such
that `M[list[i]] == i`.

See also [`gilbertindices`](@ref).
"""
function linearindices(list::Vector{CartesianIndex{2}})
    cmax = maximum(list)
    L = zeros(Int,cmax.I)
    for (i,c) in enumerate(list)
        L[c] = i
    end
    return L
end

end
