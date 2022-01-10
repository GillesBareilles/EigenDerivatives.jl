function ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    gx = g(map, x)

    E = eigvecs(gx)[:, end-eigmult.r+1:end]
    Uᵣ = U(eigmult, E)
    return Uᵣ[:, i]' * gx * Uᵣ[:, j]
end

function Dϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, d, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @warn "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    E = reverse(eigmult.Ē, dims = 2)

    return E[:, i]' * Dg(map, x, d) * E[:, j]
end

function ∇ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, i, j) where {Tf, Tm<:AbstractMap{Tf}}
    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @warn "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
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

function ∇²ϕᵢⱼ(eigmult::EigMult, map::Tm, x::Vector{Tf}, d::Vector{Tf}, i, j) where {Tf, Tm <: AbstractMap{Tf}}
    @assert eigmult.i == 1
    r = eigmult.r
    m = map.m

    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @warn "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, map, x)
    end

    gx = @timeit_debug "g oracle" g(map, x)
    η = @timeit_debug "Dg oracles" Dg(map, x, d)

    λs, E = @timeit_debug "eigen" eigen(gx)
    τ(i, k) = dot(E[:, k], η * E[:, i])

    eₗ = zeros(Tf, length(x))
    function ν(i, k, l)
        eₗ .= 0
        eₗ[l] = 1
        return dot(E[:, k], map.As[l] * E[:, i])
    end
    ♈(i) = size(gx, 1) - i + 1

    res = zeros(Tf, size(x))
    @timeit_debug "computation" for l in axes(res, 1), k in r+1:m
        scalar = 0.5 * (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)]))
        res[l] += scalar * (τ(♈(i), ♈(k)) * ν(♈(j), ♈(k), l) + ν(♈(i), ♈(k), l) * τ(♈(j), ♈(k)))
    end

    # Curvature of the map
    for l in axes(res, 1)
        res[l] += E[:, i]' * D²g_ηl(map, x, d, l) * E[:, j]
    end

    return res
end
