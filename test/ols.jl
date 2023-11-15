import GMM:
    gmm_num_residuals,
    gmm_num_instruments,
    gmm_num_parameters,
    gmm_instruments,
    gmm_residuals_constraints!,
    gmm_residuals_constraints_jacobians!

struct OLS <: GMMModel
    y::Vector{Float64}
    X::Matrix{Float64}
end

function gmm_num_residuals(ols::OLS)
    return size(ols.X, 1)
end

function gmm_num_instruments(ols::OLS)
    return size(ols.X, 2)
end

function gmm_num_parameters(ols::OLS)
    return size(ols.X, 2)
end

function gmm_instruments(ols::OLS)
    return ols.X
end

function gmm_residuals_constraints!(ols::OLS, beta, residuals, constraints)
    residuals .= ols.y .- (ols.X * beta)
end

function gmm_residuals_constraints_jacobians!(ols::OLS, beta, residuals_jacobian, constraints_jacobian)
    @. residuals_jacobian = -ols.X
end

@testset "OLS" begin
    beta = [0.25, 0.75]
    x1 = 10 .* randn(1000)
    x2 = exp.(randn(1000))
    X = [x1 x2]
    e = randn(1000)

    y = (X * beta) .+ e

    model = OLS(y, X)

    solve(model, I)
end


struct IV <: GMMModel
    y::Vector{Float64}
    X::Matrix{Float64}
    Z::Matrix{Float64}
end

function gmm_num_residuals(iv::IV)
    return size(iv.X, 1)
end

function gmm_num_instruments(iv::IV)
    return size(iv.Z, 2)
end

function gmm_num_parameters(iv::IV)
    return size(iv.X, 2)
end

function gmm_instruments(iv::IV)
    return iv.Z
end

function gmm_residuals_constraints!(iv::IV, beta, residuals, constraints)
    residuals .= iv.y .- (iv.X * beta)
end

function gmm_residuals_constraints_jacobians!(iv::IV, beta, residuals_jacobian, constraints_jacobian)
    @. residuals_jacobian = -iv.X
end

@testset "2SLS" begin
    beta = [0.25, 0.75]
    pi = [0.4, 0.6]

    N = 1_000

    z1 = 5 .* exp.(randn(N))
    Z = [ones(N) z1]
    u = randn(1000)

    x1 = (Z * pi) .+ u
    X = [ones(N) x1]
    e = randn(1000)

    y = (X * beta) .+ e

    model = IV(y, X, Z)

    solve(model, I)
end
