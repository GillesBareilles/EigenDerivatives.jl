function ∇ϕᵢⱼ(eigmult::EigMult, affmap::AffineMap, x::Vector{Tf}, i, j) where {Tf}
    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @warn "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, affmap, x)
    end

    res = zeros(Tf, size(x))
    E = reverse(eigmult.Ē, dims = 2)

    for l in axes(res, 1)
        res[l] = E[:, i]' * affmap.As[l] * E[:, j]
    end
    return res
end

function ∇²ϕᵢⱼ(eigmult::EigMult, affmap::AffineMap, x::Vector{Tf}, d::Vector{Tf}, i, j) where {Tf}
    @assert eigmult.i == 1
    r = eigmult.r
    m = affmap.m

    # The gradient can only be evaluated at the reference point. Hence, no explicit U
    if eigmult.x̄ != x
        @warn "∇ϕᵢⱼ should be evaluated at reference point. Setting it."
        update_refpoint!(eigmult, affmap, x)
    end

    gx = @timeit_debug "g oracle" g(affmap, x)
    η = @timeit_debug "Dg oracles" Dg(affmap, x, d)

    λs, E = @timeit_debug "eigen" eigen(gx)
    τ(i, k) = dot(E[:, k], η * E[:, i])
    ν(i, k, l) = dot(E[:, k], affmap.As[l] * E[:, i])
    ♈(i) = size(gx, 1) - i + 1

    res = zeros(Tf, size(x))
    @timeit_debug "computation" for l in axes(res, 1), k in r+1:m
        scalar = 0.5 * (inv(λs[♈(i)] - λs[♈(k)]) + inv(λs[♈(j)] - λs[♈(k)]))
        res[l] += scalar * (τ(♈(i), ♈(k)) * ν(♈(j), ♈(k), l) + ν(♈(i), ♈(k), l) * τ(♈(j), ♈(k)))
    end

    return res
end
