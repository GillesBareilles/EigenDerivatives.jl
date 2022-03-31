struct NonLinearMap{Tf, Tm1, Tm2} <: AbstractMap{Tf}
    n::Int64
    m::Int64
    A₀::Symmetric{Tf, Tm1}
    As::Vector{Symmetric{Tf, Tm2}}
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

function Dgconj(map::NonLinearMap{Tf}, x::Vector{Tf}, Η) where {Tf}
    res = [x[i]^(i-1) * dot(Aᵢ, Η) for (i, Aᵢ) in enumerate(map.As)]
    return res
end

function D²g(map::NonLinearMap{Tf}, x, η, ξ) where Tf
    m = map.m
    res = zeros(Tf, m, m)
    for (i, Aᵢ) in enumerate(map.As)
        res += (i-1) .* η[i] .* ξ[i] .* x[i]^(i-2) .* Aᵢ
    end
    return Symmetric(res)
end


################################################################################
#### Specialized derivatives
################################################################################
function Dg_l(map::NonLinearMap{Tf}, x, l::Int64) where Tf
    return Symmetric(x[l]^(l-1) .* map.As[l].data)
end

function D²g_ηl(map::NonLinearMap{Tf}, x, η, l::Int64) where Tf
    return Symmetric((l-1) .* η[l] .* x[l]^(l-2) .* map.As[l])
end

function D²g_kl(map::NonLinearMap{Tf}, x::Vector{Tf}, k::Int64, l::Int64) where Tf
    if k == l
        return Symmetric((l-1) * x[l]^(l-2) .* map.As[l].data)
    else
        return Tf(0) .* map.As[1].data
    end
end



################################################################################
#### Problem instances
################################################################################
function get_symmetricmat(Tf, m)
    res = rand(Tf, m, m)
    res .+= res'
    return Symmetric(res)
end


function get_nlmap(n, m; Tf = Float64, seed = 1643)
    Random.seed!(seed)
    return NonLinearMap(
        n,
        m,
        get_symmetricmat(Tf, m),
        [get_symmetricmat(Tf, m) for i in 1:n]
    )
end

function get_map()
    n = 3
    m = 7
    A = NonLinearMap(
        n,
        m,
        Symmetric(zeros(m, m)),
        [Symmetric(Diagonal(vcat(1, zeros(m-1)))) for k in 1:n]
    )
    return n, m, A

end
