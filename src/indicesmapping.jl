function ij2l(i, j, r)
    if i == j
        return ii2l(i, r)
    else
        return ij2l(i, j)
    end
end

function ii2l(i, r)
    return Int64(r * (r - 1) / 2 + i)
end

function l2ij(l::Int64, r)
    i = Int64(l - r * (r - 1) / 2)
    if i > 0
        # Diagonal elements
        return i, i
    else
        # lower triangular elements
        return l2ij(l)
    end
end

function l2ij(l::Int64)
    i::Int64 = 1
    while i * (i - 1) / 2 < l
        i += 1
    end
    j::Int64 = l - (i - 1) * ((i - 1) - 1) / 2
    return i, j
end

function ij2l(i, j)
    return Int64((i - 1) * ((i - 1) - 1) / 2 + j)
end

"""
    $TYPEDSIGNATURES

Compute the codimension of the set of matrices which largest eigenvalue has
multiplicity `r`.
"""
function hsize(r)
    return Int64(r * (r + 1) / 2 - 1)
end
function hsize(eigmult::EigMult)
    return hsize(eigmult.r)
end

l_lowerdiag(r) = 1:Int64(r * (r - 1) / 2)
l_partialdiag(r) = Int64(r * (r - 1) / 2 + 1):Int64(r * (r + 1) / 2 - 1)

function hmat2vecsmall!(hvec, hmat, r)
    # Lower diag coefficients
    for l in l_lowerdiag(r)
        i, j = l2ij(l, r)
        # @show l, i, j
        hvec[l] = hmat[i, j]
    end
    # All but last diag coefficients
    for l in l_partialdiag(r)
        i, j = l2ij(l, r)
        # @show l, i, j
        hvec[l] = hmat[i, j] - hmat[end, end]
    end
    return hvec
end

function test_indices()
    r = 5

    for i in 2:r, j in 1:r
        i <= j && continue
        @show i, j, ij2l(i, j, r)
    end

    for i in 1:(r - 1)
        @show i, ij2l(i, i, r)
    end

    println()
    println()
    for l in 1:(r * (r + 1) / 2 - 1)
        @show l, l2ij(l, r)
    end
    return nothing
end
