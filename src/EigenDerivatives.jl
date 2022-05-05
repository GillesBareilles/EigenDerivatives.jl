module EigenDerivatives

using DocStringExtensions
using LinearAlgebra
using GenericLinearAlgebra
using GenericSchur
using Manifolds

using Random
using TimerOutputs
using Infiltrator

"""
    $TYPEDSIGNATURES

An abstract type for the map `g`.
"""
abstract type AbstractMap{Tf} end

export EigMult

include("affinemap/affinemap.jl")
include("nlmap/nonlinearmap.jl")
include("powercoordmap/powercoordmap.jl")

export AffineMap, get_AL33_affinemap, get_affinemap

export NonLinearMap, get_nlmap

export PowerCoordMap, get_powercoordmap

"""
    $TYPEDSIGNATURES

Hold the multiplicity information (\$i\$-th eigenvalue is \$r\$), and the
reference basis `Ē`.

The eigenvalues are assumed to be sorted in *decreasing* order.
"""
struct EigMult{Tf,Tv<:AbstractVector{Tf}}
    i::Int64
    r::Int64
    x̄::Tv
    Ē::Matrix{Tf}
end

function EigMult(i, r, x::AbstractVector{Tf}, affmap::AffineMap{Tf}) where {Tf}
    eigmult = EigMult(i, r, x .- 1, zeros(Tf, affmap.m, r))
    update_refpoint!(eigmult, affmap, x)
    return eigmult
end
function EigMult(i, r, x::AbstractVector{Tf}, map::PowerCoordMap{Tf}) where {Tf}
    eigmult = EigMult(i, r, x .- 1, zeros(Tf, map.m, r))
    update_refpoint!(eigmult, map, x)
    return eigmult
end
function EigMult(i, r, x::AbstractVector{Tf}, nlmap::NonLinearMap{Tf}) where {Tf}
    eigmult = EigMult(i, r, x .- 1, zeros(Tf, nlmap.m, r))
    update_refpoint!(eigmult, nlmap, x)
    return eigmult
end

export g, Dg, Dgconj, D²g_ηl, D²g
export update_refpoint!
export U

"""
    $TYPEDSIGNATURES

The columns are sorted in decreasing order of corresponding eigenvalues.
"""
function update_refpoint!(
    eigmult::EigMult{Tf}, map::Tm, x̄::Vector{Tf}
) where {Tf,Tm<:AbstractMap}
    @assert eigmult.i == 1
    if eigmult.x̄ != x̄
        eigmult.x̄ .= x̄
        eigmult.Ē .= eigvecs(g(map, x̄))[:, (end - eigmult.r + 1):end]
        reverse!(eigmult.Ē; dims=2)
    else
        @debug "Not updating ref point"
    end

    return nothing
end

"""
    $TYPEDSIGNATURES

Compute a smooth basis from `E` with reference basis `Ē`. The resulting basis
is equivalent to the first one, with reversed columns.
"""
function U(eigmult::EigMult, x, gx)
    @assert eigmult.i == 1
    r = eigmult.r

    if x == eigmult.x̄
        return eigmult.Ē
    else
        E = eigvecs(gx)[:, (end - eigmult.r + 1):end]
        reverse!(E; dims=2)

        return E * project(Stiefel(r, r), E' * eigmult.Ē)
    end
end

include("phi_ij.jl")
# include("h.jl")
# include("lagrangian.jl")

include("indicesmapping.jl")
include("oraclesfullrank.jl")

export hsize
export h!, Dh!, Jacₕ!
export F̃, ∇F̃!
export Lagrangian, ∇L!, ∇²L!

# include("affinemap/phi_ij.jl")

export ϕᵢⱼ, Dϕᵢⱼ, ∇ϕᵢⱼ, ∇²ϕᵢⱼ, D²ϕᵢⱼ

end # module
