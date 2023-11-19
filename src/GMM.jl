module GMM

using LinearAlgebra: UniformScaling, issymmetric, mul!
using SparseArrays: SparseArrays, sparse, sparse_vcat, findnz
using KNITRO

include("gmmmodel.jl")
export GMMModel

include("knitro.jl")
export solve

end
