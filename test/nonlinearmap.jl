using Random
using Test
using LinearAlgebra
using PlotsOptim

@testset "NonLinearMap" begin
    @testset "g derivatives - Tf=$Tf" for Tf in [Float64, BigFloat]
        seed = 1643
        Random.seed!(seed)

        n, m = 10, 5
        A = get_nlmap(n, m;Tf, seed)

        x = rand(Tf, n)
        d = rand(Tf, n)

        Dgxd = Dg(A, x, d)
        model_to_functions = OrderedDict{String, Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "g - 1" => t -> norm(g(A, x+t*d) - g(A, x) - t*Dgxd),
        )

        for l in 1:m
            el = zeros(Tf, n)
            el[l] = 1.0

            Dgxl = Dg(A, x, el)
            D2gxl = D²g_ηl(A, x, el, l)

            model_to_functions["g - 2 - $l"] = (t -> norm(g(A, x+t*el) - g(A, x) - t*Dgxl - 0.5 * t^2 * D2gxl))
        end

        fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
        savefig(fig, "/tmp/nlfunction_$(Tf)"; savetex = false)
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "$curve" for (curve, targetslope) in union(
            [("g - 1", 2.0)],
            [("g - 2 - $l", 3.0) for l in 1:m]
        )

            slope = res[curve][1][1]
            residual = res[curve][2]
            # either the slope is correct or the model is exact, so the residual are the absciss axis.
            @test (slope >= targetslope-0.1) || (res[curve][1] == [.0, .0])
            @test residual ≈ 0.0 atol=1e-1
        end
    end


    @testset "ϕᵢⱼ derivatives - Tf=$Tf" for Tf in [
        Float64,
        # BigFloat
        ]
            seed = 1643
            Random.seed!(seed)
            n, m = 10, 5
            A = get_nlmap(n, m;Tf, seed)

            x = rand(Tf, n)
            d = rand(Tf, n)

            eigmult = EigMult(1, 2, x, A)
            update_refpoint!(eigmult, A, x)

            ϕcomp = [ϕᵢⱼ(eigmult, A, x, i, j) for i in 1:2, j in 1:2]

            φ(t) = x + t*d
            update_refpoint!(eigmult, A, x)

            i, j = 1, 1
            k, l = 1, 2
            cgradᵢⱼ = dot(∇ϕᵢⱼ(eigmult, A, x, i, j), d)
            chessᵢⱼ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, i, j), d)
            cgradₖₗ = dot(∇ϕᵢⱼ(eigmult, A, x, k, l), d)
            chessₖₗ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, k, l), d)
            model_to_functions = OrderedDict{String, Function}(
                "t" => t -> t,
                "t2" => t -> t^2,
                "t3" => t -> t^3,
                "phi i,j - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * cgradᵢⱼ),
                "phi k,l - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * cgradₖₗ),
                "phi i,j - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * cgradᵢⱼ - 0.5 * t^2 * chessᵢⱼ),
                "phi k,l - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * cgradₖₗ - 0.5 * t^2 * chessₖₗ),
            )

            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/nlfunction_phi_$(Tf)"; savetex = false)
            res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

            @testset "curve $curve" for (curve, targetslope) in [
                ("phi i,j - 1", 2.0),
                ("phi k,l - 1", 2.0),
                ("phi i,j - 2", 3.0),
                ("phi k,l - 2", 3.0),
                ]

                slope = res[curve][1][1]
                residual = res[curve][2]
                @test slope >= targetslope-0.2
                @test residual ≈ 0.0 atol=1e-1
            end

            # for (model, slopes) in res
            #     @show model, slopes
            # end
        end
    end
end