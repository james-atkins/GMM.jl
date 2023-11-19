using GMM
import GMM:
    gmm_num_residuals,
    gmm_num_instruments,
    gmm_num_constraints,
    gmm_num_parameters,
    gmm_instruments,
    gmm_residuals_constraints!,
    gmm_sparsity,
    gmm_residuals_constraints_jacobians!

using Test
import Random
using LinearAlgebra: I
using SparseArrays

Random.seed!(12345)

# include("ols.jl")
# include("iv.jl")
include("ols_sparse.jl")
