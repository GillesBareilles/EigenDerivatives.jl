"""
    $TYPEDSIGNATURES


"""
struct PowerCoordMap{Tf} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    # k::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf}}
end

function g(map::PowerCoordMap{Tf}, x::Vector{Tf}) where {Tf}
    res = copy(map.A₀.data)
    for i in 1:map.n
        res .+= x[i]^2 .* map.As[i].data
    end
    return Symmetric(res)
end

function Dg(map::PowerCoordMap{Tf}, x::Vector{Tf}, η::Vector{Tf}) where {Tf}
    res = zeros(Tf, size(first(map.As)))
    for i in 1:map.n
        res .+= 2 .* x[i] .* η[i] .* map.As[i].data
    end
    return Symmetric(res)
end

function Dgconj(map::PowerCoordMap{Tf}, x::Vector{Tf}, Η::Vector{Tf}) where {Tf}
    res = [dot(Aᵢ, Η) for Aᵢ in map.As]
    res .*= 2 .* x
    return res
end

function D²g(map::PowerCoordMap{Tf}, x, η, ξ) where Tf
    m = map.m
    res = zeros(Tf, m, m)
    for (i, Aᵢ) in enumerate(map.As)
        res += 2*η[i]*ξ[i]*Aᵢ
    end
    return Symmetric(res)
end

################################################################################
#### Specialized derivatives
################################################################################
function Dg_l(map::PowerCoordMap{Tf}, x, l::Int64) where Tf
    return Symmetric(2 .* x[l] .* map.As[l].data)
end

function D²g_ηl(map::PowerCoordMap{Tf}, x, η, l::Int64) where Tf
    return Symmetric(2 .* η[l] .* map.As[l].data)
end



function get_powercoordmap(;n = 5, m = 5, Tf = Float64)
    A₀ = Symmetric(rand(Tf, m, m))
    As = [ Symmetric(rand(Tf, m, m)) for i in 1:n ]
    return PowerCoordMap{Tf}(n, m, A₀, As)
end
