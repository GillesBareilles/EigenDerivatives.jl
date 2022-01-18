"""
    $TYPEDSIGNATURES


"""
struct PowerCoordMap{Tf, k} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf}}
end

function g(map::PowerCoordMap{Tf, k}, x::Vector{Tf}) where {Tf, k}
    res = copy(map.A₀.data)
    for i in 1:map.n
        res .+= x[i]^k .* map.As[i].data
    end
    return Symmetric(res)
end

function Dg(map::PowerCoordMap{Tf, k}, x::Vector{Tf}, η::Vector{Tf}) where {Tf, k}
    res = zeros(Tf, size(first(map.As)))
    for i in 1:map.n
        res .+= k .* x[i]^(k-1) .* η[i] .* map.As[i].data
    end
    return Symmetric(res)
end

function Dgconj(map::PowerCoordMap{Tf, k}, x::Vector{Tf}, Η) where {Tf, k}
    res = [dot(Aᵢ, Η) for Aᵢ in map.As]
    res .*= k .* x.^(k-1)
    return res
end

function D²g(map::PowerCoordMap{Tf, k}, x, η, ξ) where {Tf, k}
    m = map.m
    res = zeros(Tf, m, m)
    for (i, Aᵢ) in enumerate(map.As)
        res += k*(k-1) * x[i]^(k-2) * η[i] * ξ[i] .* Aᵢ
    end
    return Symmetric(res)
end

################################################################################
#### Specialized derivatives
################################################################################
function Dg_l(map::PowerCoordMap{Tf, k}, x, l::Int64) where {Tf, k}
    return Symmetric(k * x[l]^(k-1) .* map.As[l].data)
end

function D²g_ηl(map::PowerCoordMap{Tf, k}, x, η, l::Int64) where {Tf, k}
    return Symmetric(k*(k-1) * x[l]^(k-2) * η[l] .* map.As[l].data)
end



function get_powercoordmap(;n = 5, m = 5, k = 2, Tf = Float64)
    A₀ = Symmetric(rand(Tf, m, m))
    As = [ Symmetric(rand(Tf, m, m)) for i in 1:n ]
    return PowerCoordMap{Tf, k}(n, m, A₀, As)
end
