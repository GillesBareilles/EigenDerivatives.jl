using EigenDerivatives
using LinearAlgebra
using Test
using PlotsOptim
using Random

@testset "Affine map" begin
    @testset "Affine map derivatives" begin
        @testset "Tf=$Tf" for Tf in [Float32, Float64, BigFloat]
            A = get_AL33_affinemap(; Tf)
            @test g(A, Tf[1, 1, 1, 1]) == Tf[2 1 1; 1 0 1; 1 1 0]

            η = Tf[1, 2, 3, 4]
            @test Dg(A, Tf[1, 1, 1, 1], η) == Tf[1 2 3; 2 -1 4; 3 4 0]
        end
    end

    @testset "ϕᵢⱼ derivatives" begin
        @testset "Tf=$Tf" for Tf in [
            Float64,
            BigFloat
            ]
            @testset "$s point" for (s, x, ϕth) in [
                ("smooth", Tf[0.5, 0, 0, 0], [1.5 0.0; 0.0 0.5]),
                ("nonsmooth", Tf[parse(Tf, "1e-8"), 0, 0, 0], Tf[1 0.0; 0.0 1] + parse(Tf, "1e-8")*Diagonal{Tf}([1, -1]))
                ]

                A = get_AL33_affinemap(; Tf)
                eigmult = EigMult(1, 2, x, A)
                update_refpoint!(eigmult, A, x)

                ϕcomp = [ϕᵢⱼ(eigmult, A, x, i, j) for i in 1:2, j in 1:2]
                @test ϕcomp ≈ ϕth atol=10*eps(Tf)

                d = rand(Tf, 4)
                φ(t) = x + t*d
                update_refpoint!(eigmult, A, x)

                i, j = 1, 1
                k, l = 1, 2
                cgradᵢⱼ = dot(∇ϕᵢⱼ(eigmult, A, x, i, j), d)
                chessᵢⱼ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, i, j), d)
                cgradₖₗ = dot(∇ϕᵢⱼ(eigmult, A, x, k, l), d)
                chessₖₗ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, k, l), d)

                @test cgradᵢⱼ ≈ Dϕᵢⱼ(eigmult, A, x, d, i, j)
                @test cgradₖₗ ≈ Dϕᵢⱼ(eigmult, A, x, d, k, l)

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
                savefig(fig, "/tmp/affine_phi_$(s)_$(Tf)"; savetex = false)
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


            @testset "large pb" begin
                n, m = 5, 5
                A = get_affinemap(; n, m, Tf)
                x = rand(Tf, n)
                eigmult = EigMult(1, 3, x, A)

                d = rand(Tf, n)
                φ(t) = x + t*d
                update_refpoint!(eigmult, A, x)

                i, j = 1, 1
                k, l = 2, 3
                cgradᵢⱼ = dot(∇ϕᵢⱼ(eigmult, A, x, i, j), d)
                chessᵢⱼ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, i, j), d)
                cgradₖₗ = dot(∇ϕᵢⱼ(eigmult, A, x, k, l), d)
                chessₖₗ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, k, l), d)

                @test cgradᵢⱼ ≈ Dϕᵢⱼ(eigmult, A, x, d, i, j)
                @test cgradₖₗ ≈ Dϕᵢⱼ(eigmult, A, x, d, k, l)

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
                savefig(fig, "/tmp/affine_phi_large_$(Tf)"; savetex = false)
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

    @testset "Lagrangian derivatives" begin
        @testset "Tf=$Tf" for Tf in [
            Float64,
            BigFloat
            ]
            @testset "$s point" for (s, x) in [
                ("smooth", Tf[0.5, 0, 0, 0]),
                ("nonsmooth", Tf[parse(Tf, "1e-8"), 0, 0, 0])
                ]

                A = get_AL33_affinemap(; Tf)
                eigmult = EigMult(1, 2, x, A)
                update_refpoint!(eigmult, A, x)

                Random.seed!(1643)
                d = rand(Tf, 4)
                λ = vec(rand(Tf, 2, 2))
                φ(t) = x + t*d
                update_refpoint!(eigmult, A, x)

                cgrad = dot(∇L(eigmult, A, x, λ), d)
                chess = dot(∇²L(eigmult, A, x, λ, d), d)
                model_to_functions = OrderedDict{String, Function}(
                    "t" => t -> t,
                    "t2" => t -> t^2,
                    "t3" => t -> t^3,
                    "Lag - 1" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad),
                    "Lag - 2" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad - 0.5 * t^2 * chess),
                )

                fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
                savefig(fig, "/tmp/affine_lagrangian_$(s)_$(Tf)"; savetex = false)
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

                # for (model, slopes) in res
                #     @show model, slopes
                # end
            end
        end
    end
end
