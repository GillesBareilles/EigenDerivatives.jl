using EigenDerivatives
using LinearAlgebra
using Test
using PlotsOptim
using Random

include("genericmaptest.jl")

@testset "Power coord map" begin
    @testset "Tf = $Tf" for Tf in [
        Float64,
        # BigFloat
        ]
        @testset "k = $k" for k in [
            1,
            2,
            4
            ]
            n, m = 10, 5
            A = get_powercoordmap(; n, m, k, Tf)
            x = rand(Tf, n)
            d = rand(Tf, n)

            test_c(A, x, d, "powercoord k=$k"; print_Taylordevs = false)
            test_hinplace(A, x, d, "powercoord k=$k"; print_Taylordevs = false)
            test_Linplace(A, x, d, "powercoord k=$k"; print_Taylordevs = false)
            test_F̃(A, x, d, "powercoord k=$k"; print_Taylordevs = false)
            test_phi(A, x, d, "powercoord k=$k"; print_Taylordevs = false)
        end
    end
end
