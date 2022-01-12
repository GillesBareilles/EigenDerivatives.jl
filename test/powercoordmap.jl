using EigenDerivatives
using LinearAlgebra
using Test
using PlotsOptim
using Random

include("genericmaptest.jl")

@testset "Power coord map - Tf = $Tf" for Tf in [
    Float64,
    BigFloat
    ]
    n, m = 5, 5
    A = get_powercoordmap(; n, m, Tf)
    x = rand(Tf, n)
    d = rand(Tf, n)
    φ(t) = x + t*d

    test_map_goracles(A, x, d, n, "powercoord")
end
