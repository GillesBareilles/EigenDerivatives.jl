# Generic implementations of derivatives, likely to be quite slow
raw"""
    $TYPEDSIGNATURES

Compute the lagrangian for the problem of minimizing the average of the `i` to
`i+r-1` eigenvalues of the `map`, under the constraint that they are equal.
The multiplier for this constraint is given as `λ`. The eigenvalues indices are
carried in `eigmult`.
$$L(x, \lambda) = (1/r) \sum_{i=1}^r \lambda_i\circ A(x) + \langle \lambda, h(x)\rangle$$
where $h(x)$ is zero exactly when $\lambda_i\circ A(x) = \ldots = \lambda_{i+r-1}\circ A(x)$.
"""
function L(eigmult::EigMult, map::Tm, x::Vector{Tf}, λvec::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    res = Tf(0.0)
    r = eigmult.r
    @assert eigmult.i == 1

    λ = reshape(λvec, (r, r))

    for i in 1:r, j in 1:r
        ϕij = ϕᵢⱼ(eigmult, map, x, i, j)
        λij = λ[i, j]
        res += -λij * ϕij
        if i == j
            res += (1+tr(λ))/r * ϕij
        end
    end

    return res
end

function ∇L(eigmult::EigMult, map::Tm, x::Vector{Tf}, λvec::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    res = zeros(Tf, size(x))
    r = eigmult.r
    @assert eigmult.i == 1

    λ = reshape(λvec, (r, r))

    ∇ϕᵢⱼ_temp = similar(x)
    for i in 1:r, j in 1:r
        ∇ϕᵢⱼ_temp .= ∇ϕᵢⱼ(eigmult, map, x, i, j)
        λij = λ[i, j]
        res .+= -λij * ∇ϕᵢⱼ_temp
        if i == j
            res .+= (1+tr(λ))/r * ∇ϕᵢⱼ_temp
        end
    end

    return res
end

function ∇²L(eigmult::EigMult, map::Tm, x::Vector{Tf}, λvec::Vector{Tf}, d::Vector{Tf}) where {Tf, Tm<:AbstractMap{Tf}}
    res = zeros(Tf, size(x))
    r = eigmult.r
    @assert eigmult.i == 1

    λ = reshape(λvec, (r, r))

    ∇²ϕᵢⱼ_temp = similar(x)
    for i in 1:r, j in 1:r
        ∇²ϕᵢⱼ_temp .= ∇²ϕᵢⱼ(eigmult, map, x, d, i, j)
        λij = λ[i, j]
        res .+= -λij * ∇²ϕᵢⱼ_temp
        if i == j
            res .+= (1+tr(λ))/r * ∇²ϕᵢⱼ_temp
        end
    end

    return res
end
