function test_c(A, x::Vector{Tf}, d::Vector{Tf}, name; print_Taylordevs=false) where Tf
    φ(t) = x + t * d
    n = length(x)

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
            @test D2gxl ≈ EigenDerivatives.D²g_kl(A, x, l, l)

            model_to_functions["g - 2 - $l"] = (t -> norm(g(A, x + t * el) - g(A, x) - t * Dgxl - 0.5 * t^2 * D2gxl))
        end

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_g_$(Tf)"; savetex = false)
        end
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
end

function test_hinplace(map, x::Vector{Tf}, d::Vector{Tf}, name; print_Taylordevs=false) where Tf
    φ(t) = x + t * d

    eigmult = EigMult(1, 3, x, map)
    update_refpoint!(eigmult, map, x)
    n = length(x)
    m = hsize(eigmult)

    @testset "h, Dh, Jacₕ" begin
        cx = g(map, x)
        Dcx = Dg(map, x, d)

        hx = zeros(Tf, m);
        Dhx = zeros(Tf, m)
        Jacₕ = zeros(Tf, m, n)
        h!(hx, eigmult, x, cx)
        Dh!(Dhx, eigmult, x, Dcx)
        Jacₕ!(Jacₕ, eigmult, map, x)
        Jacₕd = Jacₕ * d

        @test rank(Jacₕ) == minimum(size(Jacₕ))

        function h(y)
            ht = zeros(Tf, m)
            h!(ht, eigmult, y, g(map, y))
            return ht
        end

        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "h differential" => t -> norm(h(φ(t)) - hx - t * Dhx),
            "h jacobian" => t -> norm(h(φ(t)) - hx - t * Jacₕd),
        )

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_hinplace_$(Tf)"; savetex = false)
        end
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("h differential", 2.0),
            ("h jacobian", 2.0),
        ]
            slope = res[curve][1][1]
            residual = res[curve][2]
            @test  slope >= targetslope - 0.2
            @test  residual ≈ 0.0 atol = 1e-1
        end
    end
end

function test_F̃(map, x::Vector{Tf}, d::Vector{Tf}, name; print_Taylordevs = false) where Tf
    φ(t) = x + t * d

    eigmult = EigMult(1, 3, x, map)
    update_refpoint!(eigmult, map, x)

    @testset "F smooth extension" begin
        cx = g(map, x)
        F̃x = F̃(eigmult, cx)
        ∇F̃x = similar(x)
        ∇F̃!(∇F̃x, eigmult, map, x)
        ∇F̃xd = dot(∇F̃x, d)

        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "F smooth extension" => t -> norm(F̃(eigmult, g(map, φ(t))) - F̃x - t * ∇F̃xd),
        )

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_Fsmoothext_$(Tf)"; savetex = false)
        end
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("F smooth extension", 2.0),
        ]
            slope = res[curve][1][1]
            residual = res[curve][2]
            @test  slope >= targetslope - 0.2
            @test  residual ≈ 0.0 atol = 1e-1
        end
    end

end

function test_Linplace(map, x::Vector{Tf}, d::Vector{Tf}, name; print_Taylordevs=false) where Tf
    φ(t) = x + t * d

    eigmult = EigMult(1, 3, x, map)
    update_refpoint!(eigmult, map, x)
    n = length(x)
    m = hsize(eigmult)

    @testset "L, ∇L!, ∇²L!" begin
        cx = g(map, x)

        λmult = rand(Tf, hsize(eigmult))

        ∇Lx = zeros(Tf, n)
        ∇²Lx = zeros(Tf, n, n)

        Lx = Lagrangian(eigmult, map, x, λmult)
        ∇L!(∇Lx, eigmult, map, x, λmult)
        ∇²L!(∇²Lx, eigmult, map, x, λmult, cx)
        ∇Lxd = dot(∇Lx, d)
        ∇²Lxd = dot(∇²Lx * d, d)

        function L(y)
            return Lagrangian(eigmult, map, y, λmult)
        end

        model_to_functions = OrderedDict{String,Function}(
            "t" => t -> t,
            "t2" => t -> t^2,
            "t3" => t -> t^3,
            "L - 1" => t -> norm(L(φ(t)) - Lx - t * ∇Lxd),
            "L - 2" => t -> norm(L(φ(t)) - Lx - t * ∇Lxd - 0.5 * t^2 * ∇²Lxd),
        )

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_Linplace_$(Tf)"; savetex = false)
        end
        res = PlotsOptim.build_affinemodels(model_to_functions; Tf)

        @testset "curve $curve" for (curve, targetslope) in [
            ("L - 1", 2.0),
            ("L - 2", 3.0),
        ]
            slope = res[curve][1][1]
            residual = res[curve][2]
            @test  slope >= targetslope - 0.2
            @test  residual ≈ 0.0 atol = 1e-1
        end
    end
end

function test_phi(A, x::Vector{Tf}, d::Vector{Tf}, name; print_Taylordevs=false) where Tf
    φ(t) = x + t * d

    eigmult = EigMult(1, 3, x, A)
    update_refpoint!(eigmult, A, x)
    # reverse!(eigmult.Ē, dims = 2)

    i, j = 1, 1
    k, l = 2, 3

    @testset "ϕᵢ - D, D²" begin
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

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_phi_differential_$(Tf)"; savetex = false)
        end
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

        if print_Taylordevs
            fig = PlotsOptim.plot_taylordev(model_to_functions; Tf)
            savefig(fig, "/tmp/$(name)_phi_gradient_$(Tf)"; savetex = false)
        end
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
end
