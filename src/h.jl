function h(eigmult::EigMult, map::Tm, x::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    r = eigmult.r
    res = zeros(Tf, r, r)
    for i in 1:r, j in 1:r
        res[i, j] = ϕᵢⱼ(eigmult, map, x, i, j)
    end
    res .-= Diagonal(tr(res)/r * ones(Tf, r))
    return vec(res)
end


function Dh(eigmult::EigMult, map::Tm, x::Vector{Tf}, η::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    r = eigmult.r
    res = zeros(Tf, r, r)
    for i in 1:r, j in 1:r
        res[i, j] = Dϕᵢⱼ(eigmult, map, x, η, i, j)
    end
    res .-= Diagonal(tr(res)/r * ones(Tf, r))
    return vec(res)
end

function Jac_h(eigmult::EigMult, map::Tm, x::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    n = length(x)
    res = zeros(Tf, eigmult.r^2, n)

    eᵢ = similar(x)
    for i in 1:n
        eᵢ .= 0
        eᵢ[i] = 1
        res[:, i] = Dh(eigmult, map, x, eᵢ)
    end

    return res
end

