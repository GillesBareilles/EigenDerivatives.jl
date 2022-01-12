using EigenDerivatives
using Random
using Test
using LinearAlgebra
using PlotsOptim


@testset "Power coord map - Tf = $Tf" for Tf in [
    Float64,
    BigFloat
    ]
    seed = 1643
    n, m = 10, 5
    A = EigenDerivatives.get_nlmap(n, m; Tf, seed)
    x = rand(Tf, n)
    d = rand(Tf, n)
    φ(t) = x + t*d

    test_map_goracles(A, x, d, n, "nlmap")
end

