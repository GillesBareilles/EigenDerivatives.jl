raw"""
    $TYPEDSIGNATURES

Compute the manifold defining map at point `x`, where $c(x)=$`cx`.
"""
function h!(res::Vector{Tf}, eigmult::EigMult{Tf}, x::Vector{Tf}, cx) where {Tf}
    E = U(eigmult, x, cx)

    hmat = E' * cx * E
    hmat2vecsmall!(res, hmat, eigmult.r)

    return res
end

raw"""
    $TYPEDSIGNATURES

Compute the differential of the manifold defining map at point `x` along a vector `d` such that `Dc(x)[d] = Dcx`.
"""
function Dh!(
    res::T, eigmult::EigMult{Tf}, x::Vector{Tf}, Dcx
) where {Tf,T<:AbstractVector{Tf}}
    @assert x == eigmult.x̄
    E = eigmult.Ē

    Dhmat = E' * Dcx * E
    hmat2vecsmall!(res, Dhmat, eigmult.r)

    return res
end

raw"""
    $TYPEDSIGNATURES

Compute the jacobian of the manifold defining map at point `x`.
"""
function Jacₕ!(
    res::T, eigmult::EigMult{Tf}, map, x::Vector{Tf}
) where {Tf,T<:AbstractMatrix{Tf}}
    for i in axes(x, 1)
        resᵢ = @view res[:, i]
        Dcxeᵢ = Dg_l(map, x, i)
        Dh!(resᵢ, eigmult, x, Dcxeᵢ)
    end
    return res
end

"""
    $TYPEDSIGNATURES

Compute the value of a smooth extension of the maximum eigenvalue relative to
the set of parameter that make the matrix have the give `eigmult`.
"""
function F̃(eigmult::EigMult{Tf}, cx) where {Tf}
    return sum(eigvals(cx)[(end - eigmult.r + 1):end]) / eigmult.r
end

"""
    $TYPEDSIGNATURES

Compute the gradient of a smooth extension of the maximum eigenvalue relative to
the set of parameter that make the matrix have the give `eigmult`.
See [`F̃`](@ref)
"""
function ∇F̃!(
    res::T, eigmult::EigMult{Tf}, map, x::Vector{Tf}
) where {Tf,T<:AbstractVector{Tf}}
    @assert x == eigmult.x̄
    E = eigmult.Ē
    res .= 0

    for l in axes(res, 1), i in 1:(eigmult.r)
        res[l] += E[:, i]' * Dg_l(map, x, l) * E[:, i]
    end
    res ./= eigmult.r
    return res
end

"""
    $TYPEDSIGNATURES

Lagrangian of maximum eigenvalue problem.

TODO
"""
function Lagrangian(eigmult::EigMult{Tf}, map, x::Vector{Tf}, λmult::Vector{Tf}) where {Tf}
    r = eigmult.r
    trλmult = sum(λmult[l_partialdiag(r)])

    res = 0.0
    for l in l_lowerdiag(r)
        i, j = l2ij(l, r)
        λᵢⱼ = λmult[l]

        res += -λᵢⱼ * ϕᵢⱼ(eigmult, map, x, i, j)
    end
    for l in l_partialdiag(r)
        i, j = l2ij(l, r)
        λᵢᵢ = λmult[l]

        res += (1 / r - λᵢᵢ) * ϕᵢⱼ(eigmult, map, x, i, j)
    end

    i = j = r
    res += (1 / r + trλmult) * ϕᵢⱼ(eigmult, map, x, i, j)
    return res
end

function ∇L!(
    res::T, eigmult::EigMult{Tf}, map, x::Vector{Tf}, λmult::Vector{Tf}
) where {Tf,T<:AbstractVector{Tf}}
    @assert x == eigmult.x̄
    r = eigmult.r
    trλmult = sum(λmult[l_partialdiag(r)])

    res .= 0
    for l in l_lowerdiag(r)
        i, j = l2ij(l, r)
        λᵢⱼ = λmult[l]

        res .+= -λᵢⱼ * ∇ϕᵢⱼ(eigmult, map, x, i, j)
    end
    for l in l_partialdiag(r)
        i, j = l2ij(l, r)
        λᵢᵢ = λmult[l]

        res .+= (1 / r - λᵢᵢ) * ∇ϕᵢⱼ(eigmult, map, x, i, j)
    end

    i = j = r
    res .+= (1 / r + trλmult) * ∇ϕᵢⱼ(eigmult, map, x, i, j)
    return res
end

"""
    $TYPEDSIGNATURES

Compute the hessian matrix corresponding to the lagrangian [`Lagrangian`](@ref).
"""
function ∇²L!(
    res::T, eigmult::EigMult{Tf}, map, x::Vector{Tf}, λmult::Vector{Tf}, cx
) where {Tf,T<:AbstractMatrix{Tf}}
    @assert x == eigmult.x̄
    r = eigmult.r
    λs, E = eigen(cx)
    res .= 0

    reverse!(λs)
    reverse!(E; dims=2)

    r = eigmult.r
    m = size(cx, 1)
    n = length(x)
    trλmult = sum(λmult[l_partialdiag(r)])

    # Precomputing coefficients
    τ = zeros(Tf, r, m, n)
    # NOTE: this is the costliest part of the function.
    for i in 1:r, s in 1:m
        for k in 1:n
            τ[i, s, k] = E[:, i]' * Dg_l(map, x, k) * E[:, s]
        end
    end

    ## Computing hessian vector product
    function ∇²ϕᵢⱼ!(res, i, j)
        res .= 0

        for k in axes(res, 1), l in axes(res, 2)
            η = zeros(Tf, n)
            η[k] = 1

            res[k, l] = E[:, i]' * D²g_kl(map, x, k, l) * E[:, j]
            for s in (r + 1):m
                scalar = 0.5 * (1 / (λs[i] - λs[s]) + 1 / (λs[j] - λs[s]))
                res[k, l] += scalar * (τ[i, s, k] * τ[j, s, l] + τ[i, s, l] * τ[j, s, k])
            end
        end
    end

    temp = zeros(Tf, n, n)
    for l in l_lowerdiag(r)
        i, j = l2ij(l, r)
        λᵢⱼ = λmult[l]

        ∇²ϕᵢⱼ!(temp, i, j)
        res .+= -λᵢⱼ * temp
    end
    for l in l_partialdiag(r)
        i, j = l2ij(l, r)
        λᵢᵢ = λmult[l]

        ∇²ϕᵢⱼ!(temp, i, j)
        res .+= (1 / r - λᵢᵢ) * temp
    end
    i = j = r

    ∇²ϕᵢⱼ!(temp, i, j)
    res .+= (1 / r + trλmult) * temp

    return res
end
