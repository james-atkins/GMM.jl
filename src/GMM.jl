module GMM

using LinearAlgebra: UniformScaling, issymmetric, mul!
using KNITRO
using SparseArrays: sparse

include("gmmmodel.jl")
export GMMModel,
    GMMResult,
    gmm_success,
    gmm_objective_value,
    gmm_estimate,
    gmm_moments,
    gmm_moments_jacobian,
    gmm_constraints_jacobian

include("knitro.jl")
export solve

end
