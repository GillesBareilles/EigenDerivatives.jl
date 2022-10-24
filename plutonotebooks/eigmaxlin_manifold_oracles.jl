### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ c8c01ff8-0971-11ec-0ca5-3504f63e6217
begin
    using Pkg
    Pkg.activate(".")

    using Revise
    using DarkMode
    using NonSmoothProblems
    using Random
    using DataStructures
    using LinearAlgebra
    using PlotsOptim

    const NSP = NonSmoothProblems
end

# ╔═╡ 928b0203-6197-43e2-a9ab-03a466f1349a
DarkMode.enable()

# ╔═╡ d53f5d57-b11e-4aba-941d-ae923db54cb7

# ╔═╡ f4d6f991-bf03-4580-8b2c-b96a8cfdd874
begin
    n = 50
    m = 25
    r = 5
    pb = get_eigmaxlinear_pb(; m=m, n=n, seed=1864)

    x̄ = [
        0.1727508879661212,
        0.14427406629905412,
        0.4804972103061302,
        -0.0017620535516393615,
        -0.18002651383772347,
        0.10428460347132448,
        -0.045063268359047885,
        -0.7252123575907988,
        0.44240309909775,
        -0.46356409076899624,
        0.2593867690625561,
        0.80899957770808,
        0.041225072834493004,
        -0.41032265346249275,
        0.6349357758961202,
        -0.13762202500460347,
        0.3645405375966907,
        -0.44339866948252776,
        -0.5494462582449624,
        -0.02191557752589415,
        0.30895202040563985,
        0.05546265135785551,
        0.11755211324290663,
        -0.4286313362337124,
        -0.13207097878262436,
        0.13711525957226717,
        -0.4595618364258303,
        0.2319054551895379,
        -0.0837268063148705,
        -0.005320801359365086,
        -0.6336428111728033,
        -0.38788346485226743,
        -0.7791171347363044,
        -0.9394872052094738,
        -0.0686980501432224,
        0.4746112627115909,
        -0.21624278368912606,
        -0.5284977157582383,
        0.3331214367943889,
        -0.5904637505226982,
        -0.7386750430407496,
        -0.45356334376036017,
        -0.23245211625938994,
        -0.43450612729771615,
        -0.06710890450824919,
        0.15756633480718324,
        0.04822123622323272,
        0.27125987046203226,
        0.15411714925026676,
        -0.10422806044730497,
    ]
    M = NSP.EigmaxLinearManifold(pb, r)

    Random.seed!(1439)
    x = rand(n)
    d = rand(n)
end

# ╔═╡ 6105e146-e286-435a-82de-87121ef0b83c
function plot(x)
    i, j = 1, 1
    k, l = 1, 2

    φ(t) = x + t * d
    M.xref .= x
    model_to_functions = OrderedDict{String,Function}(
        "t" => t -> t,
        "t2" => t -> t^2,
        "t3" => t -> t^3,
        "phi i,j - 2" =>
            t -> norm(
                NSP.ϕᵢⱼ(M, φ(t), i, j) - NSP.ϕᵢⱼ(M, φ(0), i, j) -
                t * NSP.Dϕᵢⱼ(M, φ(0), i, j, d) -
                0.5 * t^2 * dot(NSP.∇²ϕᵢⱼ(M, φ(0), i, j, d), d),
            ),
        "phi k,l - 2" =>
            t -> norm(
                NSP.ϕᵢⱼ(M, φ(t), k, l) - NSP.ϕᵢⱼ(M, φ(0), k, l) -
                t * NSP.Dϕᵢⱼ(M, φ(0), k, l, d) -
                0.5 * t^2 * dot(NSP.∇²ϕᵢⱼ(M, φ(0), k, l, d), d),
            ),
        "U - 1" => t -> norm(NSP.U(M, φ(t)) - NSP.U(M, φ(0)) - t * NSP.DU(M, φ(0), d)),
    )

    return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ 278ca712-f87b-4679-91f4-4c52400c2dca

# ╔═╡ 4f57c6f6-980f-4fa9-8ad0-f9eeefb8db00
plot(x)

# ╔═╡ 494246cc-4e84-41e9-8784-d058a9eae556
plot(x̄)

# ╔═╡ 93099891-5a31-4351-aa37-82c9958233e0
x

# ╔═╡ 19199ef5-f099-42cd-8235-79aa03fbe05e
md"""
### losange
Checking 

$dU_i . dA . U_j$
"""

# ╔═╡ 390ef614-c289-4d00-83fe-b153d3123db4
begin
    i, j = 2, 1
    NSP.duᵢdAuⱼ(M, x, d, i, j)
end

# ╔═╡ 67ac9c8d-dd16-40e2-8aab-7594e3930740
NSP.DU(M, x, d)' * NSP.Dg(pb, x, d) * NSP.U(M, x)

# ╔═╡ b58dd4df-7eb9-4826-ac44-97434041d4c6
md"""
### carré
Checking 

$dU_i . dA . U_j$
"""

# ╔═╡ 14944ca0-de86-4ef3-8fab-a1a3efbe0d41
function aa(M::EigmaxLinearManifold, x, d, i, j)
    gx = NSP.g(M.pb, x)
    η = NSP.Dg(M.pb, x, d)
    res = zeros(size(gx, 1), M.r)

    λs, E = eigen(gx)

    j_flip = size(gx, 1) - j + 1

    res = zeros(size(gx, 1), M.r)
    for i in 1:(M.r), k in (M.r + 1):(M.pb.m)
        i_flip = size(gx, 1) - i + 1
        k_flip = size(gx, 1) - k + 1
        scalar = inv(λs[i_flip] - λs[k_flip]) * dot(E[:, k_flip], η * E[:, i_flip])
        res[:, M.r - i + 1] .+= scalar .* E[:, k_flip]
    end

    i_flip = size(gx, 1) - i + 1
    res = res[:, i]' * η * E[:, i_flip]

    return res
end

# ╔═╡ 06af361d-5a9d-4adf-b5f7-e6242dcd1dd3
aa(M, x, d, i, j)

# ╔═╡ 54c89dc6-6846-4e76-8fa5-be911667eb39
NSP.U(M, x)

# ╔═╡ 697626d8-0417-4365-a16c-88811a724a78
gx = NSP.g(pb, x);

# ╔═╡ 40f65f77-d97f-45b1-a573-4ab5b9696420
eigvecs(gx)[:, (end - r + 1):end]

# ╔═╡ Cell order:
# ╠═c8c01ff8-0971-11ec-0ca5-3504f63e6217
# ╠═928b0203-6197-43e2-a9ab-03a466f1349a
# ╠═d53f5d57-b11e-4aba-941d-ae923db54cb7
# ╠═f4d6f991-bf03-4580-8b2c-b96a8cfdd874
# ╠═6105e146-e286-435a-82de-87121ef0b83c
# ╠═278ca712-f87b-4679-91f4-4c52400c2dca
# ╠═4f57c6f6-980f-4fa9-8ad0-f9eeefb8db00
# ╠═494246cc-4e84-41e9-8784-d058a9eae556
# ╠═93099891-5a31-4351-aa37-82c9958233e0
# ╠═19199ef5-f099-42cd-8235-79aa03fbe05e
# ╠═390ef614-c289-4d00-83fe-b153d3123db4
# ╠═67ac9c8d-dd16-40e2-8aab-7594e3930740
# ╠═b58dd4df-7eb9-4826-ac44-97434041d4c6
# ╠═14944ca0-de86-4ef3-8fab-a1a3efbe0d41
# ╠═06af361d-5a9d-4adf-b5f7-e6242dcd1dd3
# ╠═40f65f77-d97f-45b1-a573-4ab5b9696420
# ╠═54c89dc6-6846-4e76-8fa5-be911667eb39
# ╠═697626d8-0417-4365-a16c-88811a724a78
