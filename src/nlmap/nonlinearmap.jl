struct NonLinearMap{Tf} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    A₀::Symmetric{Tf}
    As::Vector{Symmetric{Tf}}
end
function g(nlmap::NonLinearMap{Tf}, x::Vector{Tf}) where {Tf}
    res = copy(nlmap.A₀.data)
    for i in 1:nlmap.n
        res .+= (x[i]^i / i) .* nlmap.As[i].data
    end
    return Symmetric(res)
end
function Dg(nlmap::NonLinearMap{Tf}, x::Vector{Tf}, η::Vector{Tf}) where {Tf}
    res = zeros(Tf, size(first(nlmap.As)))
    for i in 1:nlmap.n
        res .+= x[i]^(i-1) .* η[i] .* nlmap.As[i].data
    end
    return Symmetric(res)
end
function D²g_ηl(nlmap::NonLinearMap{Tf}, x::Vector{Tf}, η::Vector{Tf}, l::Int64) where {Tf}
    return Symmetric(x[l]^(l-2) .* (l-1) .* η[l] .* nlmap.As[l])
end


function get_symmetricmat(Tf, m)
    res = rand(Tf, m, m)
    res .+= res'
    return Symmetric(res)
end


function get_nlmap(n, m; Tf = Float64, seed = 1643)
    Random.seed!(seed)
    return NonLinearMap{Tf}(
        n,
        m,
        get_symmetricmat(Tf, m),
        [get_symmetricmat(Tf, m) for i in 1:n]
    )
end
