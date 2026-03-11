module GilbertCurves

export gilbertindices

"""
    gilbertindices(dims::Tuple{Int,Int}; majdim=dims[1] >= dims[2] ? 1 : 2)

Constructs a vector of `CartesianIndex` objects, orderd by a generalized Hilbert
space-filling, starting at `CartesianIndex(1,1)`.  It will end at
`CartesianIndex(dims[1],1)` if `majdim==1`, or `CartesianIndex(1,dims[2])` if `majdim==2`
(or the closest feasible point).
"""
gilbertindices(dims::Tuple; kwargs...) =
    gilbertorder(CartesianIndices(dims); kwargs...)


"""
    gilbertorder(mat::AbstractMatrix; majdim=size(mat,1) >= size(mat,2) ? 1 : 2)

Constructs a vector of the elements of `mat`, ordered by a generalized Hilbert
space-filling curve. The list will start at `mat[1,1]`, and end at `mat[end,1]` if
`majdim==1` or `mat[1,end]` if `majdim==2`  (or the closest feasible point).
"""
function gilbertorder(mat::AbstractMatrix{T}; majdim=size(mat,1) >= size(mat,2) ? 1 : 2) where {T}
    list = sizehint!(T[], length(mat))
    if majdim == 1
        append_gilbert!(list, mat)
    else
        append_gilbert!(list, permutedims(mat,(2,1)))
    end
end

# 3D ordering
function gilbertorder(mat::AbstractArray{T, 3}; majdim=findmax(size(mat))) where {T}
    list = sizehint!(T[], length(mat))
    
    if majdim == 1
        append_gilbert!(list, mat)
    elseif majdim == 2
        append_gilbert!(list, permutedims(mat,(2,1,3)))
    else
        append_gilbert!(list, permutedims(mat,(3,2,1)))
    end
end

# 2D ordering
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

# 3D ordering
function append_gilbert!(list, mat::AbstractArray{T, 3}) where {T}
    a,b,c = size(mat)

    if sum((a, b, c) .== 1) > 1
        # single in one dimension
        append!(list, mat)
    # wide case, split in width only
    elseif (2a > 3b) && (2a > 3c)
        a2 = div(a,2)
        if isodd(a2) && a > 2
            a2 += 1
        end
        append_gilbert!(list, mat[1:a2,:,:])
        append_gilbert!(list, mat[a2+1:a,:,:])
    # do not split in depth
    elseif 3b > 4c
        a2 = div(a,2)
        b2 = div(b,2)
        if isodd(b2) && b > 2
            b2 += 1
        end
        append_gilbert!(list, permutedims(mat[1:a2,1:b2,:],(2,3,1)))
        append_gilbert!(list, mat[:,b2+1:b,:])
        append_gilbert!(list, permutedims(mat[a:-1:a2+1,b2:-1:1,:],(2,3,1)))
    # do not split in height
    elseif 3c > 4b
        a2 = div(a,2)
        c2 = div(c,2)
        if isodd(a2) && a > 2
            a2 += 1
        end
        if isodd(c2) && c > 2
            c2 += 1
        end
        append_gilbert!(list, permutedims(mat[1:a2,:,1:c2],(3,1,2)))
        append_gilbert!(list, mat[:,:,c2+1:c])
        append_gilbert!(list, permutedims(mat[a:-1:a2+1,:,c2:-1:1],(3,1,2)))
    # regular case, split in all w/h/d 
    else
        a2 = div(a,2)
        b2 = div(b,2)
        c2 = div(c,2)
        if isodd(a2) && a > 2
            a2 += 1
        end
        if isodd(b2) && b > 2
            b2 += 1
        end
        if isodd(c2) && c > 2
            c2 += 1
        end
        append_gilbert!(list, permutedims(mat[1:a2,1:b2,1:c2],(2,3,1))) 
        append_gilbert!(list, permutedims(mat[1:a2,b2+1:b,:],(3,1,2))) 
        append_gilbert!(list, mat[:,b2:-1:1,c:-1:c2+1])
        append_gilbert!(list, permutedims(mat[a:-1:a2+1,b2+1:b,c:-1:1],(3,1,2)))
        append_gilbert!(list, permutedims(mat[a:-1:a2+1,b2:-1:1,1:c2],(2,3,1)))
    end
end

"""
    linearindices(list::Vector{CartesianIndex{2}})

Construct an integer Array `M` containing the integers `1:length(list)` such that
`M[list[i]] == i`.
"""
function linearindices(list::Vector{CartesianIndex{N}}) where {N}
    cmax = maximum(list)
    L = zeros(Int,cmax.I)
    for (i,c) in enumerate(list)
        L[c] = i
    end
    return L
end

end