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

export AffineMap, get_AL33_affinemap
export NonLinearMap, get_nlmap

"""
    $TYPEDSIGNATURES

Hold the multiplicity information (\$i\$-th eigenvalue is \$r\$), and the
reference basis `Ē`.

The eigenvalues are assumed to be sorted in *decreasing* order.
"""
struct EigMult{Tf, Tv<:AbstractVector{Tf}}
    i::Int64
    r::Int64
    x̄::Tv
    Ē::Matrix{Tf}
end

function EigMult(i, r, x::AbstractVector{Tf}, affmap::AffineMap{Tf}) where Tf
    eigmult = EigMult(i, r, x .- 1, zeros(Tf, affmap.m, r))
    update_refpoint!(eigmult, affmap, x)
    return eigmult
end
function EigMult(i, r, x::AbstractVector{Tf}, nlmap::NonLinearMap{Tf}) where Tf
    eigmult = EigMult(i, r, x .- 1, zeros(Tf, nlmap.m, r))
    update_refpoint!(eigmult, nlmap, x)
    return eigmult
end



"""
    $TYPEDSIGNATURES

Compute the eigenvalues and eigenvectors of `A` at indices `eigenvals_indices`.
"""
function get_eigendecomp(A::AbstractMatrix, eigenvals_indices::UnitRange{Int64})
    λs, E = eigen(A)
    return λs[eigenvals_indices], E[eigenvals_indices]
end


"""
    $TYPEDSIGNATURES

Compute a smooth basis from `E` with reference basis `Ē`. The resulting basis
is equivalent to the first one, with reversed columns. In particular, the
columns are sorted in decreasing order of corresponding eigenvalues.
"""
function U(coalescenceinds, E, Ē)
    # NOTE: is this working for sthg other than max eigenval?
    res = E * project(Stiefel(M.r, M.r), E' * Ē)
    reverse!(res, dims=2)
    return res
end




export g, Dg, Dgconj, D²g_ηl
export update_refpoint!
export U


############################
function update_refpoint!(eigmult::EigMult{Tf}, map::Tm, x̄::Vector{Tf}) where {Tf, Tm<:AbstractMap}
    @assert eigmult.i == 1
    if eigmult.x̄ != x̄
        eigmult.x̄ .= x̄
        eigmult.Ē .= eigvecs(g(map, x̄))[:, end-eigmult.r+1:end]
    else
        @debug "Not updating ref point"
    end

    return nothing
end


function U(eigmult::EigMult, E) where Tf
    r = eigmult.r
    @assert eigmult.i == 1

    res = E * project(Stiefel(r, r), E' * eigmult.Ē)
    reverse!(res, dims=2)
    return res
end


include("phi_ij.jl")
include("h.jl")
include("lagrangian.jl")

include("affinemap/phi_ij.jl")

export ϕᵢⱼ, Dϕᵢⱼ, ∇ϕᵢⱼ, ∇²ϕᵢⱼ
export h, Dh, Jac_h
export L, ∇L, ∇²L


end # module
