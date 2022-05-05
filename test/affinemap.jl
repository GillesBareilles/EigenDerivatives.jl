using EigenDerivatives
using LinearAlgebra
using Test
using PlotsOptim

include("genericmaptest.jl")

@testset "Affine map" begin
    @testset "Affine map derivatives" begin
        @testset "Tf=$Tf" for Tf in [Float32, Float64, BigFloat]
            A = get_AL33_affinemap(; Tf)
            @test g(A, Tf[1, 1, 1, 1]) == Tf[2 1 1; 1 0 1; 1 1 0]

            η = Tf[1, 2, 3, 4]
            @test Dg(A, Tf[1, 1, 1, 1], η) == Tf[1 2 3; 2 -1 4; 3 4 0]
        end
    end

    @testset "Tf = $Tf" for Tf in [Float64, BigFloat]
        n, m = 10, 5
        A = get_affinemap(n, m; Tf)
        x = rand(Tf, n)
        d = rand(Tf, n)

        test_c(A, x, d, "nonlinearmap"; print_Taylordevs=false)
        test_hinplace(A, x, d, "nonlinearmap"; print_Taylordevs=false)
        test_Linplace(A, x, d, "nonlinearmap"; print_Taylordevs=false)
        test_F̃(A, x, d, "nonlinearmap"; print_Taylordevs=false)
        test_phi(A, x, d, "nonlinearmap"; print_Taylordevs=false)
    end
end
