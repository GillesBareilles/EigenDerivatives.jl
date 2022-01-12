function ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    gx = g(map, x)

    E = eigvecs(gx)[:, end-eigmult.r+1:end]
    Uᵣ = U(eigmult, E)
    return Uᵣ[:, i]' * gx * Uᵣ[:, j]
end
function Dϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, d, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @debug "Dϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end
    E = reverse(eigmult.Ē, dims = 2)

    return E[:, i]' * Dg(map, x, d) * E[:, j]
end
function D²ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, η::Vector{Tf}, ξ::Vector{Tf}, i, j) where {Tm, Tf}
    @assert eigmult.i == 1
    r = eigmult.r
    m = map.m

    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @debug "D²ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    gx = @timeit_debug "g oracle" g(map, x)
    λ, E = @timeit_debug "eigen" eigen(gx)
    reverse!(E, dims=2)
    reverse!(λ)
    τ(i, k, η) = E[:, i]' * Dg(map, x, η) * E[:, k]

    res = E[:, i]' * D²g(map, x, η, ξ) * E[:, j]
    for k in r+1:m
        scalar = 0.5 * (1/(λ[i] - λ[k]) + 1/(λ[j] - λ[k]))
        res += scalar * (τ(i, k, η)*τ(j, k, ξ) + τ(i, k, ξ) * τ(j, k, η))
    end

    return res
end




function ∇ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @debug "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    res = zeros(Tf, size(x))
    E = reverse(eigmult.Ē, dims = 2)

    eₗ = zeros(Tf, length(x))
    for l in axes(res, 1)
        eₗ .= 0
        eₗ[l] = 1
        res[l] = E[:, i]' * Dg(map, x, eₗ) * E[:, j]
    end
    return res
end

function ∇²ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, η::Vector{Tf}, i, j) where {Tf, Tm <: AbstractMap{Tf}}
    @assert eigmult.i == 1
    r = eigmult.r
    n, m = map.n, map.m

    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @debug "∇²ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    gx = @timeit_debug "g oracle" g(map, x)
    λ, E = @timeit_debug "eigen" eigen(gx)
    reverse!(E, dims=2)
    reverse!(λ)
    τ(i, k, η) = E[:, i]' * Dg(map, x, η) * E[:, k]
    ν(i, k, l) = E[:, i]' * Dg_l(map, x, l) * E[:, k]

    res = similar(x)
    for l in 1:n
        res[l] = E[:, i]' * D²g_ηl(map, x, η, l) * E[:, j]
        for k in r+1:m
            scalar = 0.5 * (1/(λ[i] - λ[k]) + 1/(λ[j] - λ[k]))
            res[l] += scalar * (τ(i, k, η)*ν(j, k, l) + ν(i, k, l) * τ(j, k, η))
        end
    end

    return res
end




