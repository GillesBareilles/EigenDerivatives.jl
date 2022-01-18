### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ d29aae10-b978-11eb-1ebd-270ca1941c45
begin
	import Pkg
	Pkg.activate()
	
	using Revise
	
	using NonSmoothProblems
	#using NonSmoothSolvers
	using PlotsOptim
	using LinearAlgebra
	using Test
	using Random
	using DataStructures
	
	using ArnoldiMethod
	using Arpack
	using IterativeSolvers
	
	using Manifolds
	using Manopt
	
	const NSP = NonSmoothProblems
	import NonSmoothProblems: g, Dg, EigmaxLinearManifold
end

# ╔═╡ b1f861f4-6e4f-4e16-9ee4-b554afdd50a4
md"""
This notebook discusses the manifolds relative to which $\lambda_\max \circ A(\cdot)$ is partly-smooth, namely

$M_r = \{x : \lambda_\max\circ A \text{ has multiplicity } r \}.$

This manifold is defined as $h^{-1}(\{0\})$ for the map $h : \mathbb{R}^n \to \mathbb{R}^{r\times r}$,

$h(x) = U(x)^\top A(x) U(x) - \frac{1}{r} \mathrm{tr} (U(x)^\top A(x) U(x)) I_r,$

and $U(x) \in \mathbb{R}^{m \times r}$ is an orthonormal basis of the eigenspace associated with the $r$ largest eigenvalues.

Plan:
- construction of a $\mathcal{C}^2$ map $U$, formulas for $U$, $DU$;
- first and second derivatives for $\phi_{i, j}(x) = U_i(x)^\top A(x) U_j(x)$;
- first and second derivatives for $h$.

References:
- *On Eigenvalue Optimization*, Shapiro and Fan (1995)
- *Perturbation Theory for Linear Operators*, Kato (1995)

## Smooth basis of space of $r$ largest eigenvalues

Denote $E(x) \in \mathbb{R}^{m\times r}$ the eigenvectors of the $r$ largest eigenvalues of $A(x)$. This function is not continuous (possible sign flips and rotations).

Let $\bar{x}$ denote a reference point and define $U(x)$ as (_On eigenvalue optimization_, p. 6)

$\arg\min \|G - \bar{E}\|^2_F, \quad G\in\mathcal M(x) \triangleq\{G\in\mathbb R^{n\times r} : G^\top G=I_r, \quad P(x)G = G\},$

where $\bar{E}=E(\bar{x})$, $P(x) = E(x)E(x)^\top$ is the orthogonal projection on the eigenspace of interest and is the original smooth object.

With the change of variables $G = E(x)H$, 

$U(x) = E(x) \arg\min_{H\in\mathcal O_r} \|E(x) H - \bar{E}\|^2 = E(x) \cdot \mathrm{proj}_{\mathcal O_r}(E(x)^\perp \bar{E}).$

The differential writes, for $1 \le i \le r$,

$dU_i(\bar{x}) = \sum_{k > r} \frac{1}{\lambda_i - \lambda_k} E_k E_k^\top dA(\bar{x}) E_i.$
"""

# ╔═╡ 2bfef944-863d-4176-9724-fb64275d2c38
function U(M::EigmaxLinearManifold, x)
    gx = NSP.g(M.pb, x)
	E = eigvecs(gx)[:, end-M.r+1:end]
	Ē = eigvecs(NSP.g(M.pb, M.xref))[:, end-M.r+1:end]
    res = E * project(Stiefel(M.r, M.r), E' * Ē)
    reverse!(res, dims=2)
	return res
end

# ╔═╡ 3cc25e76-a22e-4ef9-acb0-0cbcd451cd41
function DU(M::EigmaxLinearManifold, x, d)
    gx = g(M.pb, x)
    η = Dg(M.pb, x, d)
	res = zeros(size(gx, 1), M.r)

    λs, E = eigen(gx)
    τ(i, k) = dot(E[:, k], η * E[:, i])
    ♈(i) = size(gx, 1) - i + 1

    for i in 1:M.r, k in M.r+1:M.pb.m
        res[:, i] .+= inv(λs[♈(i)] - λs[♈(k)]) * τ(♈(i), ♈(k)) .* E[:, ♈(k)]
    end
	return res
end

# ╔═╡ 32f85698-94be-4b42-8d77-36e9e73fb304
md"""
> Checking the differential at two points: $x$ away from $M$ and $\bar{x}$ close to the manifold.

> /!\ Beware that the reference point should be updated each time a derivative is computed.
"""

# ╔═╡ 4a1952d0-e60d-4042-9106-7bf50cbfdbb7
md"""
## Derivatives of $U_i(x)^\top A(x) U_j(x)$

Let $\phi_{i, j}(x) = U_i(x)^\top A(x) U_j(x)$. Following Kato, Shapiro and Fan, we get:

$D\phi_{i, j}(\bar{x})[\eta] = U_i(\bar{x})^\top dA(\bar{x})[\eta] U_j(\bar{x})$

$D^2\phi_{i, j}(\bar{x})[\eta, \xi] = E_i^\top D^2A(\bar{x})[\eta, \xi] E_j + \sum_{k > r} \frac{1}{2} \left(\frac{1}{\lambda_i - \lambda_k} + \frac{1}{\lambda_j - \lambda_k} \right) \left\{ \tau_{i, k}^\eta \tau_{j, k}^\xi + \tau_{i, k}^\xi \tau_{j, k}^\eta \right\},$
where $\tau_{i, k}^\eta = E_i^\top DA(\bar{x})[\eta] E_j$.

Below shows the case $\eta = \xi$ and $A$ affine.
"""

# ╔═╡ 8d8832b5-ee88-4164-b663-97d82199adba
begin
	function ϕ(M::EigmaxLinearManifold, x)
	    gx = g(M.pb, x)
	    Uᵣ = U(M, x)
	    return Uᵣ' * gx * Uᵣ
	end
	
	function Dϕ(M::EigmaxLinearManifold, x, d)
	    Dgx = Dg(M.pb, x, d)
	    Uᵣ = U(M, x)
	    return Uᵣ' * Dgx * Uᵣ
	end
	
	ϕᵢⱼ(M::EigmaxLinearManifold, x, i, j) = ϕ(M, x)[i, j]
	Dϕᵢⱼ(M::EigmaxLinearManifold, x, d, i, j) = Dϕ(M, x, d)[i, j]
		
	function DDϕᵢⱼ(M::EigmaxLinearManifold, x, d, i, j)
		gx = g(M.pb, x)
		η = Dg(M.pb, x, d)

		λs, E = eigen(gx)
		τ(i, k) = dot(E[:, k], η * E[:, i])
		♈(i) = size(gx, 1) - i + 1

		res = 0.0
		for k in M.r+1:M.pb.m
			res += (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)])) * τ(♈(i), ♈(k)) * τ(♈(j), ♈(k))
		end

		return res
	end
end

# ╔═╡ 5d02a822-5d8a-4f1b-8a3e-ddf6f2730602
md"""
Optimization wise, a different representation of these tensors is more useful. When $A(x) = A_0 + \sum_{l = 1}^n x_l A_l$,

$\nabla \phi_{i, j}(\bar{x})_l = U_i(\bar{x})^\top A_l U_j(\bar{x})$

$\nabla^2\phi_{i, j}(\bar{x})[\eta]_l = \sum_{k > r} \frac{1}{2} \left(\frac{1}{\lambda_i - \lambda_k} + \frac{1}{\lambda_j - \lambda_k} \right) \left\{ \tau_{i, k}^\eta \tau_{j, k}^l + \tau_{i, k}^l \tau_{j, k}^\eta \right\},$
where $\tau_{i, k}^\eta = E_i^\top DA(\bar{x})[\eta] E_j$ and $\tau_{i, k}^l = E_i^\top DA(\bar{x})[e_i] E_j$.

"""

# ╔═╡ 0bca980a-9d6e-4eb7-aa7a-47944fb6115d
function ∇ϕᵢⱼ(M::EigmaxLinearManifold, x, i, j)
	res = zeros(size(x))
	Uᵣ = U(M, x)
	
	for l in axes(res, 1)
		res[l] = Uᵣ[:, i]' * M.pb.As[l+1] * Uᵣ[:, j]
	end
	return res
end

# ╔═╡ b2a5b20d-81c2-4fcc-8f39-2b7075c54463
function ∇²ϕᵢⱼ(M::EigmaxLinearManifold, x, d, i, j)
    gx = NSP.g(M.pb, x)
    η = NSP.Dg(M.pb, x, d)

    λs, E = eigen(gx)
    τ(i, k) = dot(E[:, k], η * E[:, i])
    ν(i, k, l) = dot(E[:, k], M.pb.As[l+1] * E[:, i])
    ♈(i) = size(gx, 1) - i + 1

	res = zeros(size(x))
    for l in axes(res, 1), k in M.r+1:M.pb.m
        scalar = 0.5 * (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)]))
        res[l] += scalar * (τ(♈(i), ♈(k)) * ν(♈(j), ♈(k), l) + ν(♈(i), ♈(k), l) * τ(♈(j), ♈(k)))
    end

	return res
end

# ╔═╡ a52890ed-1fc1-48fa-bcaf-35552be15da8
md"""
## Derivatives of $h$

We are now able to define the manifold of matrices which largest eigenvalue has a fixed multiplicty with function $h$:

$h(x) = U(x)^\top A(x) U(x) - \frac{1}{r} \mathrm{tr} (U(x)^\top A(x) U(x)) I_r$.

Its gradient writes:

$Dh(\bar{x})[\eta] = U(\bar{x})^\top dA(\bar{x})[\eta] U(\bar{x}) - \frac{1}{r} \mathrm{tr} (U(\bar{x})^\top dA(\bar{x})[\eta] U(\bar{x})) I_r$

Its hessian writes:

$\nabla^2\phi_{i, j}(\bar{x})[\eta]_l = \sum_{k > r} \frac{1}{2} \left(\frac{1}{\lambda_i - \lambda_k} + \frac{1}{\lambda_j - \lambda_k} \right) \left\{ \tau_{i, k}^\eta \tau_{j, k}^l + \tau_{i, k}^l \tau_{j, k}^\eta \right\},$
"""

# ╔═╡ a0eba8f7-f9f8-494b-879e-87e925076574
md"""
## Lagrangian hessian
We are finally able to form the lagrangian hessian, in a naive and quite inefficient way first.
"""

# ╔═╡ ab4c8b54-ec6d-47ea-a89a-044f29337fcb
function lagrangian(pb, M::EigmaxLinearManifold, x, λ)
	res = ϕᵢⱼ(M, x, 1, 1)
	
	for i in 1:M.r, j in 1:M.r
		res += λ[i, j] * ϕᵢⱼ(M, x, i, j)
	end
	return res
end

# ╔═╡ 9bee6ae4-5ba1-4c1c-9237-d80d771104d9
function ∇lagrangian(pb, M::EigmaxLinearManifold, x, λ)
	res = ∇ϕᵢⱼ(M, x, 1, 1)
	
	for i in 1:M.r, j in 1:M.r
		res .+= λ[i, j] .* ∇ϕᵢⱼ(M, x, i, j)
	end
	return res
end

# ╔═╡ 488fc162-fa65-43fd-863b-fcd03acc2c14
function ∇²lagrangian(pb, M::EigmaxLinearManifold, x, λ, η)
	res = ∇²ϕᵢⱼ(M, x, η, 1, 1)
	
	for i in 1:M.r, j in 1:M.r
		res .+= λ[i, j] .* ∇²ϕᵢⱼ(M, x, η, i, j)
	end
	return res
end

# ╔═╡ 8c1e5483-1ddd-42f9-bcc3-aaf097113a09
begin
	function build_σζ(M, E, m, n, r, η)
	    σ = zeros(n, r, m)
	    for l in 1:n, i in 1:r, k in 1:m
	        σ[l, i, k] = dot(E[:, m-k+1], M.pb.As[l+1] * E[:, m-i+1])
	    end
	    ζ = zeros(r, m)
	    for i in 1:r, k in 1:m
	        ζ[i, k] = dot(E[:, m-k+1], η* E[:, m-i+1])
	    end
	    return σ, ζ
	end
	
	function fill_res!(res, M, λs, λmat, σ, ζ)
	    i = j = 1
	    for l in axes(res, 1), k in M.r+1:M.pb.m
	        scalar = 0.5 * (inv(λs[i] - λs[k]) + inv(λs[j] - λs[k]))
	        res[l] += scalar * (ζ[i, k] * σ[l, j, k] + σ[l, i, k] * ζ[j, k])
	    end
	
	    for i in 1:M.r, j in 1:M.r
	        for l in axes(res, 1), k in M.r+1:M.pb.m
	            scalar = λmat[i, j] * 0.5 * (inv(λs[i] - λs[k]) + inv(λs[j] - λs[k]))
	            res[l] -= scalar * (ζ[i, k] * σ[l, j, k] + σ[l, i, k] * ζ[j, k])
	        end
	    end
	    return
	end
	
	function ∇²Lagrangian_improved(pb::EigmaxLinear, M::EigmaxLinearManifold, x::Vector{Tf}, λ, d) where {Tf}
	    M.xref .= x
	    gx = NSP.g(M.pb, x)::Symmetric{Tf, Matrix{Tf}}
	    η = NSP.Dg(M.pb, x, d)::Symmetric{Tf, Matrix{Tf}}
	    λs, E = eigen(gx)
	
	    reverse!(λs)
	
	    n, m = pb.n, pb.m
	    r = M.r
	
	    σ, ζ = build_σζ(M, E, m, n, r, η)
	
	    λmat = reshape(λ, (M.r, M.r))
	    res = zeros(size(x))
	    ## obj hessian
	    fill_res!(res, M, λs, λmat, σ, ζ)
	
	    return res
	end
end

# ╔═╡ 95a70d5a-4196-4721-9dfe-0ca9c067d8bd
md"""
## Annex
"""

# ╔═╡ 5e598b88-0189-4f22-a1da-58420e006346
begin
	n = 50
	m = 25
	r = 5
	pb = get_eigmaxlinear_pb(m=m, n=n, seed = 1864)
		
	x̄ = [0.1727508879661212, 0.14427406629905412, 0.4804972103061302, -0.0017620535516393615, -0.18002651383772347, 0.10428460347132448, -0.045063268359047885, -0.7252123575907988, 0.44240309909775, -0.46356409076899624, 0.2593867690625561, 0.80899957770808, 0.041225072834493004, -0.41032265346249275, 0.6349357758961202, -0.13762202500460347, 0.3645405375966907, -0.44339866948252776, -0.5494462582449624, -0.02191557752589415, 0.30895202040563985, 0.05546265135785551, 0.11755211324290663, -0.4286313362337124, -0.13207097878262436, 0.13711525957226717, -0.4595618364258303, 0.2319054551895379, -0.0837268063148705, -0.005320801359365086, -0.6336428111728033, -0.38788346485226743, -0.7791171347363044, -0.9394872052094738, -0.0686980501432224, 0.4746112627115909, -0.21624278368912606, -0.5284977157582383, 0.3331214367943889, -0.5904637505226982, -0.7386750430407496, -0.45356334376036017, -0.23245211625938994, -0.43450612729771615, -0.06710890450824919, 0.15756633480718324, 0.04822123622323272, 0.27125987046203226, 0.15411714925026676, -0.10422806044730497]
	M = NSP.EigmaxLinearManifold(pb, r)
	
	Random.seed!(1439)
	x = rand(n)
	d = rand(n)
	
	i, j = 1, 1
	k, l = 1, 2
	
	λ = rand(r, r)
end

# ╔═╡ 72ffb700-40f6-4b5e-8aa7-9fe83bace680
function plot_U(x)
    φ(t) = x + t*d
	M.xref .= x
	model_to_functions = OrderedDict{String, Function}(
		"t" => t -> t,
		"t2" => t -> t^2,
		"t3" => t -> t^3,
		"U" => t -> norm(U(M, φ(t)) - U(M, φ(0)) - t * DU(M, φ(0), d)),
	)
	
	return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ 9c15d27f-2693-4220-81de-1e0c85e74967
eigvals(NSP.g(pb, x))[end-r:end]

# ╔═╡ ee4b440f-49e8-4c93-8faf-87feb2d3e440
eigvals(NSP.g(pb, x̄))[end-r:end]

# ╔═╡ 9e0039ca-8f64-46f0-8ce2-23a54dc4882a
plot_U(x)

# ╔═╡ a057bcc3-54b8-4efe-bb90-c560802bffad
plot_U(x̄)

# ╔═╡ 9f17a80e-b134-4447-b98d-8615328fa737
function plot_ϕ(x)
	φ(t) = x + t*d
	M.xref .= x
	model_to_functions = OrderedDict{String, Function}(
		"t" => t -> t,
		"t2" => t -> t^2,
		"t3" => t -> t^3,
		"phi i,j - 2" => t -> norm(ϕᵢⱼ(M, φ(t), i, j) - ϕᵢⱼ(M, φ(0), i, j) - t * Dϕᵢⱼ(M, φ(0), d, i, j) - 0.5 * t^2 * DDϕᵢⱼ(M, φ(0), d, i, j)),
		"phi k,l - 2" => t -> norm(ϕᵢⱼ(M, φ(t), k, l) - ϕᵢⱼ(M, φ(0), k, l) - t * Dϕᵢⱼ(M, φ(0), d, k, l) - 0.5 * t^2 * DDϕᵢⱼ(M, φ(0), d, k, l)),
	)
	
	return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ 81447a3d-8f57-433f-bab5-202463624535
plot_ϕ(x)

# ╔═╡ a9902777-c3f8-4427-8c45-4951f0f3b610
plot_ϕ(x̄)

# ╔═╡ 7fc5aeb9-c1b3-46f6-83f2-24d73318afcc
function plot_ϕ2(x)
	M.xref .= x
    φ(t) = x + t*d
	
	cgradᵢⱼ = dot(∇ϕᵢⱼ(M, x, i, j), d)
	chessᵢⱼ = dot(∇²ϕᵢⱼ(M, x, d, i, j), d)
	cgradₖₗ = dot(∇ϕᵢⱼ(M, x, k, l), d)
	chessₖₗ = dot(∇²ϕᵢⱼ(M, x, d, k, l), d)
	model_to_functions = OrderedDict{String, Function}(
		"t" => t -> t,
		"t2" => t -> t^2,
		"t3" => t -> t^3,
		"phi i,j - 2" => t -> norm(ϕᵢⱼ(M, φ(t), i, j) - ϕᵢⱼ(M, φ(0), i, j) - t * cgradᵢⱼ - 0.5 * t^2 * chessᵢⱼ),
		"phi k,l - 2" => t -> norm(ϕᵢⱼ(M, φ(t), k, l) - ϕᵢⱼ(M, φ(0), k, l) - t * cgradₖₗ - 0.5 * t^2 * chessₖₗ),
	)
	
	return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ b4568afd-2cca-445d-97ea-f92371c75a3a
plot_ϕ2(x)

# ╔═╡ c266466f-b42f-4b85-b10c-2c82922bf3df
plot_ϕ2(x̄)

# ╔═╡ 096b5798-2f47-4e15-b482-4a4b8eb89762
begin
	function h(M::EigmaxLinearManifold, x)
	    gx = g(M.pb, x)
	    Uᵣ = U(M, x)
	
	    res = Uᵣ' * gx * Uᵣ
	    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))
	    return vec(res)
	end
	
	function Dh(M::EigmaxLinearManifold, x, d)
		gx = g(M.pb, x)
	    Dgx = Dg(M.pb, x, d)
		Uᵣ = U(M, x)
	
	    res = Uᵣ' * Dgx * Uᵣ
	    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))
	    return vec(res)
	end
	
	function hᵢⱼ(M::EigmaxLinearManifold, x, i, j)
		gx = g(M.pb, x)
	    Uᵣ = U(M, x)
	
	    res = Uᵣ' * gx * Uᵣ
	    res .-= tr(res) ./ M.r .* Diagonal(ones(M.r))
		return res[i, j]
	end
	
	function ∇hᵢⱼ(M::EigmaxLinearManifold, x, i, j)
		res = ∇ϕᵢⱼ(M, x, i, j)
		
		if i == j
			for l in 1:M.r
				res .-= ∇ϕᵢⱼ(M, x, l, l) ./ r
			end
		end
	    return res
	end
	
	function ∇²hᵢⱼ(M::EigmaxLinearManifold, x, d, i, j)
		res = ∇²ϕᵢⱼ(M, x, d, i, j)
		
		if i == j
			for l in 1:M.r
				res .-= ∇²ϕᵢⱼ(M, x, d, l, l) ./ r
			end
		end
	    return res
	end
end

# ╔═╡ f91be01c-efa7-443d-8d5a-323444a36d4c
function plot_h(x)
	M.xref .= x
    φ(t) = x + t*d
	
	cgradᵢⱼ = dot(∇hᵢⱼ(M, x, i, j), d)
	chessᵢⱼ = dot(∇²hᵢⱼ(M, x, d, i, j), d)
	cgradₖₗ = dot(∇hᵢⱼ(M, x, k, l), d)
	chessₖₗ = dot(∇²hᵢⱼ(M, x, d, k, l), d)
	model_to_functions = OrderedDict{String, Function}(
		"t" => t -> t,
		"t2" => t -> t^2,
		"t3" => t -> t^3,
		"h" => t -> norm(h(M, φ(t)) - h(M, φ(0)) - t * Dh(M, φ(0), d)),
		"h ij" => t -> norm(hᵢⱼ(M, φ(t), i, j) - hᵢⱼ(M, φ(0), i, j) - t * cgradᵢⱼ - 0.5 * t^2 * chessᵢⱼ),
		"h kl" => t -> norm(hᵢⱼ(M, φ(t), k, l) - hᵢⱼ(M, φ(0), k, l) - t * cgradₖₗ - 0.5 * t^2 * chessₖₗ),
	)
	
	return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ 4ca410ea-60b5-4b76-ad92-1c93a1e30fb7
plot_h(x)

# ╔═╡ 58af110b-0430-41a5-a18f-f4bd5bf147d3
plot_h(x̄)

# ╔═╡ c7cc55dc-fd39-4c1f-8051-a68ac6170155
function plot_lagrangian(x, λ)
	M.xref .= x
    φ(t) = x + t*d
	
	cgrad = dot(∇lagrangian(pb, M, x, λ), d)
	chess = dot(∇²lagrangian(pb, M, x, λ, d), d)
	chess_i = dot(∇²Lagrangian_improved(pb, M, x, λ, d), d)
	model_to_functions = OrderedDict{String, Function}(
		"t" => t -> t,
		"t2" => t -> t^2,
		"t3" => t -> t^3,
		"lagrangian" => t -> norm(lagrangian(pb, M, φ(t), λ) - lagrangian(pb, M, φ(0), λ) - t * cgrad - 0.5 * t^2 * chess),
		"lag fast" => t -> norm(lagrangian(pb, M, φ(t), λ) - lagrangian(pb, M, φ(0), λ) - t * cgrad - 0.5 * t^2 * chess),
	)
	
	return PlotsOptim.plot_curves(build_logcurves(model_to_functions))
end

# ╔═╡ 7c6686a5-e396-4dc7-8931-5f31ff5b6c1d
plot_lagrangian(x, λ)

# ╔═╡ a6e7b985-592c-4a59-ad22-0af06bad1942
plot_lagrangian(x̄, λ)

# ╔═╡ e7bca459-0c5d-4843-9e2d-1bbefda78b20
begin
    import DarkMode
    DarkMode.enable()
    #DarkMode.Toolbox(theme="default")
end

# ╔═╡ Cell order:
# ╟─b1f861f4-6e4f-4e16-9ee4-b554afdd50a4
# ╠═2bfef944-863d-4176-9724-fb64275d2c38
# ╠═3cc25e76-a22e-4ef9-acb0-0cbcd451cd41
# ╠═72ffb700-40f6-4b5e-8aa7-9fe83bace680
# ╟─32f85698-94be-4b42-8d77-36e9e73fb304
# ╠═9c15d27f-2693-4220-81de-1e0c85e74967
# ╠═ee4b440f-49e8-4c93-8faf-87feb2d3e440
# ╠═9e0039ca-8f64-46f0-8ce2-23a54dc4882a
# ╠═a057bcc3-54b8-4efe-bb90-c560802bffad
# ╟─4a1952d0-e60d-4042-9106-7bf50cbfdbb7
# ╠═8d8832b5-ee88-4164-b663-97d82199adba
# ╠═9f17a80e-b134-4447-b98d-8615328fa737
# ╠═81447a3d-8f57-433f-bab5-202463624535
# ╠═a9902777-c3f8-4427-8c45-4951f0f3b610
# ╟─5d02a822-5d8a-4f1b-8a3e-ddf6f2730602
# ╠═0bca980a-9d6e-4eb7-aa7a-47944fb6115d
# ╠═b2a5b20d-81c2-4fcc-8f39-2b7075c54463
# ╠═7fc5aeb9-c1b3-46f6-83f2-24d73318afcc
# ╠═b4568afd-2cca-445d-97ea-f92371c75a3a
# ╠═c266466f-b42f-4b85-b10c-2c82922bf3df
# ╟─a52890ed-1fc1-48fa-bcaf-35552be15da8
# ╠═096b5798-2f47-4e15-b482-4a4b8eb89762
# ╠═f91be01c-efa7-443d-8d5a-323444a36d4c
# ╠═4ca410ea-60b5-4b76-ad92-1c93a1e30fb7
# ╠═58af110b-0430-41a5-a18f-f4bd5bf147d3
# ╟─a0eba8f7-f9f8-494b-879e-87e925076574
# ╠═ab4c8b54-ec6d-47ea-a89a-044f29337fcb
# ╠═9bee6ae4-5ba1-4c1c-9237-d80d771104d9
# ╠═488fc162-fa65-43fd-863b-fcd03acc2c14
# ╠═c7cc55dc-fd39-4c1f-8051-a68ac6170155
# ╠═7c6686a5-e396-4dc7-8931-5f31ff5b6c1d
# ╠═a6e7b985-592c-4a59-ad22-0af06bad1942
# ╠═8c1e5483-1ddd-42f9-bcc3-aaf097113a09
# ╟─95a70d5a-4196-4721-9dfe-0ca9c067d8bd
# ╠═5e598b88-0189-4f22-a1da-58420e006346
# ╠═d29aae10-b978-11eb-1ebd-270ca1941c45
# ╠═e7bca459-0c5d-4843-9e2d-1bbefda78b20
