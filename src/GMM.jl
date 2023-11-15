module GMM

using LinearAlgebra: UniformScaling, issymmetric, mul!
using KNITRO

include("gmmmodel.jl")
export GMMModel

include("knitro.jl")
export solve

end
