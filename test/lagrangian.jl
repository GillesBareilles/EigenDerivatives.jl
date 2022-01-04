using Random
using Test
using LinearAlgebra
using PlotsOptim

@testset "Lagrangian derivatives" begin
    @testset "Tf=$Tf" for Tf in [
        Float64,
        # BigFloat
        ]
        @testset "$s point" for (s, x) in [
            ("smooth", Tf[0.5, 0, 0, 0]),
            # ("nonsmooth", Tf[parse(Tf, "1e-8"), 0, 0, 0])
            ]

            A₀ = [1 0 0; 0 1 0; 0 0 0]
            As = [
                [1 0 0; 0 -1 0; 0 0 0],
                [0 1 0; 1 0 0; 0 0 0],
                [0 0 1; 0 0 0; 1 0 0],
                [0 0 0; 0 0 1; 0 1 0],
            ]
            A = NonLinearMap{Tf}(4, 3, Symmetric(Tf.(A₀)), [Symmetric(Tf.(a)) for a in As])
            eigmult = EigMult(1, 2, x, A)
            update_refpoint!(eigmult, A, x)

            Random.seed!(1643)
            d = rand(Tf, 4)
            λ = rand(Tf, 2, 2)
            φ(t) = x + t*d
            update_refpoint!(eigmult, A, x)

            cgrad = dot(∇L(eigmult, A, x, λ), d)
            chess = dot(∇²L(eigmult, A, x, λ, d), d)
            @show chess
            @show d
            model_to_functions = OrderedDict{String, Function}(
                "t" => t -> t,
                "t2" => t -> t^2,
                "t3" => t -> t^3,
                "Lag - 1" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad),
                "Lag - 2" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad - 0.5 * t^2 * chess),
            )

            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/lagrangian_$(s)_$(Tf)"; savetex = false)
            res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

            @testset "curve $curve" for (curve, targetslope) in [
                ("Lag - 1", 2.0),
                ("Lag - 2", 3.0),
                ]

                slope = res[curve][1][1]
                residual = res[curve][2]
                @test slope >= targetslope-0.1
                @test residual ≈ 0.0 atol=1e-1
            end
        end
    end
end
