"""
    $TYPEDSIGNATURES


"""
struct AffineMap{Tf} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf, Matrix{Tf}}}
end

function g(map::AffineMap{Tf}, x::Tv) where {Tf,Tv<:AbstractVector{Tf}}
    res = zeros(Tf, map.m, map.m)
    res .= map.A₀
    for i in 1:(map.n)
        res .+= x[i] .* map.As[i].data
    end
    return Symmetric(res)
end

function Dg(map::AffineMap{Tf}, x::Tv, η::Vector{Tf}) where {Tf,Tv<:AbstractVector{Tf}}
    res = zeros(Tf, size(first(map.As)))
    for i in 1:(map.n)
        res .+= η[i] .* map.As[i].data
    end
    return Symmetric(res)
end

function Dgconj(map::AffineMap{Tf}, x::Tv, ξ) where {Tf,Tv<:AbstractVector{Tf}}
    res = [dot(ξ, Aᵢ) for Aᵢ in map.As]
    return res
end

function D²g(map::AffineMap{Tf}, x, η, ξ) where {Tf}
    return zeros(Tf, size(map.A₀))
end

function D²g_kl(
    map::AffineMap{Tf}, x::Tv, k::Int64, l::Int64
) where {Tf,Tv<:AbstractVector{Tf}}
    return Diagonal(Tf(0) * I, size(map.As[1], 1))
end

################################################################################
#### Specialized derivatives
################################################################################
function Dg_l(map::AffineMap{Tf}, x, l::Int64) where {Tf}
    return Symmetric(map.As[l].data)
end

function D²g_ηl(map::AffineMap{Tf}, x, η, l::Int64) where {Tf}
    return zeros(Tf, size(map.A₀))
end

function get_AL33_affinemap(; Tf=Float64)
    A₀ = [1 0 0; 0 1 0; 0 0 0]
    As = [
        [1 0 0; 0 -1 0; 0 0 0],
        [0 1 0; 1 0 0; 0 0 0],
        [0 0 1; 0 0 0; 1 0 0],
        [0 0 0; 0 0 1; 0 1 0],
    ]
    return AffineMap{Tf}(4, 3, Symmetric(Tf.(A₀)), [Symmetric(Tf.(a)) for a in As])
end

function get_affinemap(n=5, m=5; Tf=Float64)
    A₀ = Symmetric(rand(Tf, m, m))
    As = [Symmetric(rand(Tf, m, m)) for i in 1:n]
    return AffineMap{Tf}(n, m, A₀, As)
end
