"""
    $TYPEDSIGNATURES


"""
struct AffineMap{Tf} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf}}
end

function g(affmap::AffineMap{Tf}, x::Vector{Tf}) where {Tf}
    res = copy(affmap.A₀.data)
    for i in 1:affmap.n
        res .+= x[i] .* affmap.As[i].data
    end
    return Symmetric(res)
end

function Dg(affmap::AffineMap{Tf}, x::Vector{Tf}, η::Vector{Tf}) where {Tf}
    res = zeros(Tf, size(first(affmap.As)))
    for i in 1:affmap.n
        res .+= η[i] .* affmap.As[i].data
    end
    return Symmetric(res)
end

function Dgconj(affmap::AffineMap{Tf}, x::Vector{Tf}, ξ::Symmetric{Tf}) where {Tf}
    res = similar(x)
    for (i, Aᵢ) in enumerate(affmap.As)
        res[i] = dot(ξ, Aᵢ)
    end
    return res
end

function D²g_ηl(affmap::AffineMap{Tf}, x, η, l::Int64) where Tf
    return zeros(Tf, length(x))
end


function get_AL33_affinemap(;Tf = Float64)
    A₀ = [1 0 0; 0 1 0; 0 0 0]
    As = [
        [1 0 0; 0 -1 0; 0 0 0],
        [0 1 0; 1 0 0; 0 0 0],
        [0 0 1; 0 0 0; 1 0 0],
        [0 0 0; 0 0 1; 0 1 0],
    ]
    return AffineMap{Tf}(4, 3, Symmetric(Tf.(A₀)), [Symmetric(Tf.(a)) for a in As])
end
