function ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf,Tm<:AbstractMap{Tf}}
    gx = g(map, x)
    Uᵣ = U(eigmult, x, gx)
    return Uᵣ[:, i]' * gx * Uᵣ[:, j]
end
function Dϕᵢⱼ(
    eigmult::EigMult, map::Tm, x::Vector{Tf}, d, i, j
) where {Tf,Tm<:AbstractMap{Tf}}
    @assert eigmult.i == 1
    if eigmult.x̄ != x
        @debug "Dϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end
    E = eigmult.Ē

    return E[:, i]' * Dg(map, x, d) * E[:, j]
end
function D²ϕᵢⱼ(
    eigmult::EigMult, map::Tm, x::Vector{Tf}, η::Vector{Tf}, ξ::Vector{Tf}, i, j
) where {Tm,Tf}
    @assert eigmult.i == 1
    r = eigmult.r
    m = map.m

    if eigmult.x̄ != x
        @debug "D²ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    gx::Matrix{Tf} = g(map, x)
    λ::Vector{Tf}, E::Matrix{Tf} = eigen(gx)
    reverse!(E; dims=2)
    reverse!(λ)
    τ(i, k, η) = E[:, i]' * Dg(map, x, η) * E[:, k]

    res::Tf = E[:, i]' * D²g(map, x, η, ξ) * E[:, j]
    for k in (r + 1):m
        scalar = 0.5 * (1 / (λ[i] - λ[k]) + 1 / (λ[j] - λ[k]))
        res += scalar * (τ(i, k, η) * τ(j, k, ξ) + τ(i, k, ξ) * τ(j, k, η))
    end

    return res
end

function ∇ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf,Tm<:AbstractMap{Tf}}
    @assert eigmult.i == 1
    if eigmult.x̄ != x
        @debug "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    res = zeros(Tf, size(x))
    E = eigmult.Ē

    eₗ = zeros(Tf, length(x))
    for l in axes(res, 1)
        eₗ .= 0
        eₗ[l] = 1
        res[l] = E[:, i]' * Dg(map, x, eₗ) * E[:, j]
    end
    return res
end

function ∇²ϕᵢⱼ(
    eigmult::EigMult, map::Tm, x::Vector{Tf}, η::Vector{Tf}, i, j
) where {Tf,Tm<:AbstractMap{Tf}}
    @assert eigmult.i == 1
    r = eigmult.r
    n, m = map.n, map.m

    if eigmult.x̄ != x
        @debug "∇²ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    gx = g(map, x)
    λ, E = eigen(gx)
    reverse!(E; dims=2)
    reverse!(λ)
    τ(i, k, η) = E[:, i]' * Dg(map, x, η) * E[:, k]
    ν(i, k, l) = E[:, i]' * Dg_l(map, x, l) * E[:, k]

    res = similar(x)
    for l in 1:n
        res[l] = E[:, i]' * D²g_ηl(map, x, η, l) * E[:, j]
        for k in (r + 1):m
            scalar = 0.5 * (1 / (λ[i] - λ[k]) + 1 / (λ[j] - λ[k]))
            res[l] += scalar * (τ(i, k, η) * ν(j, k, l) + ν(i, k, l) * τ(j, k, η))
        end
    end

    return res
end
