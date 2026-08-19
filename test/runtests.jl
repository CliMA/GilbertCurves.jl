using GilbertCurves
using Test

@testset "size $m,$n" for m = 1:20, n = 1:20
    list = gilbertindices((m,n))
    @test length(list) == m*n
    @test list[1] == CartesianIndex(1,1)

    # The curve ends at the far corner of the major dimension, except for
    # odd-by-2 grids, where checkerboard parity makes that endpoint
    # unreachable and the curve ends elsewhere.
    if m >= n
        if !(n == 2 && isodd(m))
            @test list[end] == CartesianIndex(m,1)
        end
    else
        if !(m == 2 && isodd(n))
            @test list[end] == CartesianIndex(1,n)
        end
    end

    ndiag = 0
    for i = 1:m*n-1
        Δ = map(abs, (list[i+1] - list[i]).I)
        @test Δ[1] <= 1
        @test Δ[2] <= 1
        if Δ[1] + Δ[2] > 1
            ndiag += 1
        end
    end

    # When the larger dimension is odd and the smaller even, no orthogonal
    # corner-to-corner path exists (checkerboard parity), so the curve
    # contains one diagonal step — except when the smaller dimension is 2,
    # where the endpoint is relaxed instead and the curve stays continuous.
    if (m > n && isodd(m) && iseven(n)) || (m < n && isodd(n) && iseven(m))
        @test ndiag == (min(m,n) == 2 ? 0 : 1)
    else
        @test ndiag == 0
    end

    # Every cell is visited exactly once
    L = GilbertCurves.linearindices(list)
    @test !any(iszero, L)
    @test sum(L) == sum(1:m*n)
end

@testset "gilbertorder matches gilbertindices" begin
    M = reshape(1:9, 3, 3)
    ordered_vals = GilbertCurves.gilbertorder(M)
    @test length(ordered_vals) == 9
    @test ordered_vals[1] == M[1,1]
    @test ordered_vals == [M[idx] for idx in gilbertindices((3,3))]
end

@testset "empty domains" begin
    # The recursion has no base case for empty blocks, so `gilbertorder`
    # must return early on an empty matrix.
    @test isempty(gilbertindices((0,0)))
    @test isempty(gilbertindices((0,5)))
    @test isempty(gilbertindices((5,0)))
end

@testset "explicit majdim override" begin
    # For a 3x5 grid the default is majdim=2
    default_inds = gilbertindices((3, 5))
    forced_inds = gilbertindices((3, 5); majdim=1)

    @test length(default_inds) == 15
    @test length(forced_inds) == 15
    @test sort(default_inds) == sort(forced_inds)
    @test default_inds != forced_inds

    @test default_inds[end] == CartesianIndex(1, 5)
    @test forced_inds[end] == CartesianIndex(3, 1)
end
