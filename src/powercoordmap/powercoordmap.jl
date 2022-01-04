"""
    $TYPEDSIGNATURES


"""
struct PowerCoordMap{Tf} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    k::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf}}
end

function g(map::PowerCoordMap{Tf}, x::Vector{Tf}) where {Tf}
    res = copy(map.A₀.data)
    for i in 1:map.n
        res .+= (1/map.k) .* x[i]^map.k .* map.As[i].data
    end
    return Symmetric(res)
end

function Dg(map::PowerCoordMap{Tf}, x::Vector{Tf}, η::Vector{Tf}) where {Tf}
    res = zeros(Tf, size(first(map.As)))
    for i in 1:map.n
        res .+= x[i]^(map.k-1) .* η[i] .* map.As[i].data
    end
    return Symmetric(res)
end

function D²g_ηl(map::PowerCoordMap{Tf}, x, η, l::Int64) where Tf
    @assert false
    return zeros(Tf, length(x))
end
