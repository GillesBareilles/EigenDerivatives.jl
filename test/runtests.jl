using Test
using EigenDerivatives
using LinearAlgebra
using PlotsOptim
using Random

include("genericmaptest.jl")

function runtests()
    include("test/affinemap.jl")
    include("test/powercoordmap.jl")
    include("test/nonlinearmap.jl")
    return
end
