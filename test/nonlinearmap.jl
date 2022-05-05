using EigenDerivatives
using Test
using LinearAlgebra
using PlotsOptim

@testset "Power coord map" begin
    @testset "Tf = $Tf" for Tf in [Float64]#, BigFloat]
        @testset "repeat: $rnd" for rnd in 1:5
            Random.seed!(1346 + rnd)
            n, m = 10, 5
            A = get_nlmap(n, m; Tf)
            x = rand(Tf, n)
            d = rand(Tf, n)

            test_c(A, x, d, "nonlinearmap"; print_Taylordevs=false)
            test_hinplace(A, x, d, "nonlinearmap"; print_Taylordevs=false)
            test_Linplace(A, x, d, "nonlinearmap"; print_Taylordevs=false)
            test_F̃(A, x, d, "nonlinearmap"; print_Taylordevs=false)
            test_phi(A, x, d, "nonlinearmap"; print_Taylordevs=false)
        end
    end
end
