function test_map_goracles(A, x::Vector{Tf}, d::Vector{Tf}, n, name) where {Tf}
    φ(t) = x + t * d

    @testset "g" begin
        Dgxd = Dg(A, x, d)
        D²gxd = D²g(A, x, d, d)
        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "g - 1" => t -> norm(g(A, x + t * d) - g(A, x) - t * Dgxd),
            "g - 2 - d" => t -> norm(g(A, φ(t)) - g(A, x) - t * Dgxd - 0.5 * t^2 * D²gxd)
        )

        for l in 1:n
            el = zeros(Tf, n)
            el[l] = 1.0

            Dgxl = Dg(A, x, el)
            D2gxl = D²g_ηl(A, x, el, l)

            model_to_functions["g - 2 - $l"] = (t -> norm(g(A, x + t * el) - g(A, x) - t * Dgxl - 0.5 * t^2 * D2gxl))
        end

        fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
        savefig(fig, "/tmp/$(name)_g_$(Tf)"; savetex = false)
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "$curve" for (curve, targetslope) in union(
            [("g - 1", 2.0)],
            [("g - 2 - $l", 3.0) for l in 1:n],
            [("g - 2 - d", 3.0)],
        )

            slope = res[curve][1][1]
            residual = res[curve][2]
            # either the slope is correct or the model is exact, so the residual are the absciss axis.
            @test (slope >= targetslope - 0.1) || (res[curve][1] == [0.0, 0.0])
            @test residual ≈ 0.0 atol = 1e-1
        end
    end

    @testset "ϕᵢ - D, D²" begin
        eigmult = EigMult(1, 3, x, A)
        update_refpoint!(eigmult, A, x)

        i, j = 1, 1
        k, l = 2, 3
        dϕᵢⱼ = Dϕᵢⱼ(eigmult, A, x, d, i, j)
        d²ϕᵢⱼ = D²ϕᵢⱼ(eigmult, A, x, d, d, i, j)
        dϕₖₗ = Dϕᵢⱼ(eigmult, A, x, d, k, l)
        d²ϕₖₗ = D²ϕᵢⱼ(eigmult, A, x, d, d, k, l)

        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "phi i,j - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * dϕᵢⱼ),
            "phi k,l - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * dϕₖₗ),
            "phi i,j - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * dϕᵢⱼ - 0.5 * t^2 * d²ϕᵢⱼ),
            "phi k,l - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * dϕₖₗ - 0.5 * t^2 * d²ϕₖₗ),
        )

        fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
        savefig(fig, "/tmp/$(name)_phi_differential_$(Tf)"; savetex = false)
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("phi i,j - 1", 2.0),
            ("phi k,l - 1", 2.0),
            ("phi i,j - 2", 3.0),
            ("phi k,l - 2", 3.0),
        ]

            slope = res[curve][1][1]
            residual = res[curve][2]
            @test slope >= targetslope - 0.2
            @test residual ≈ 0.0 atol = 1e-1
        end
    end

    @testset "ϕᵢⱼ - gradient, hessian" begin
        eigmult = EigMult(1, 3, x, A)
        update_refpoint!(eigmult, A, x)

        i, j = 1, 1
        k, l = 2, 3
        cgradᵢⱼ = dot(∇ϕᵢⱼ(eigmult, A, x, i, j), d)
        chessᵢⱼ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, i, j), d)
        cgradₖₗ = dot(∇ϕᵢⱼ(eigmult, A, x, k, l), d)
        chessₖₗ = dot(∇²ϕᵢⱼ(eigmult, A, x, d, k, l), d)

        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "phi i,j - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * cgradᵢⱼ),
            "phi k,l - 1" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * cgradₖₗ),
            "phi i,j - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), i, j) - ϕᵢⱼ(eigmult, A, φ(0), i, j) - t * cgradᵢⱼ - 0.5 * t^2 * chessᵢⱼ),
            "phi k,l - 2" => t -> norm(ϕᵢⱼ(eigmult, A, φ(t), k, l) - ϕᵢⱼ(eigmult, A, φ(0), k, l) - t * cgradₖₗ - 0.5 * t^2 * chessₖₗ),
        )

        fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
        savefig(fig, "/tmp/$(name)_phi_gradient_$(Tf)"; savetex = false)
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("phi i,j - 1", 2.0),
            ("phi k,l - 1", 2.0),
            ("phi i,j - 2", 3.0),
            ("phi k,l - 2", 3.0),
        ]

            slope = res[curve][1][1]
            residual = res[curve][2]
            @test slope >= targetslope - 0.2
            @test residual ≈ 0.0 atol = 1e-1
        end
    end

    @testset "Lagrangian - gradient, hessian" begin
        r = 3
        eigmult = EigMult(1, r, x, A)
        update_refpoint!(eigmult, A, x)

        λ = rand(Tf, r*r)

        cgrad = dot(∇L(eigmult, A, x, λ), d)
        chess = dot(∇²L(eigmult, A, x, λ, d), d)
        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "Lag - 1" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad),
            "Lag - 2" => t -> norm(L(eigmult, A, φ(t), λ) - L(eigmult, A, x, λ) - t * cgrad - 0.5 * t^2 * chess),
        )

        fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
        savefig(fig, "/tmp/$(name)_lagrangian_$(Tf)"; savetex = false)
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("Lag - 1", 2.0),
            ("Lag - 2", 3.0),
        ]

            slope = res[curve][1][1]
            residual = res[curve][2]
            @test slope >= targetslope - 0.1
            @test residual ≈ 0.0 atol = 1e-1
        end
    end
    return
end
