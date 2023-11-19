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
    N = 1_000
    beta = [0.25, 0.75]
    x1 = 10 .* randn(N)
    x2 = exp.(randn(N))
    X = [x1 x2]
    e = randn(N)

    y = (X * beta) .+ e
    beta_hat = (X' * X) \ (X' * y)

    model = OLS(y, X)
    _, beta_hat_gmm = solve(model, I)

    @test isapprox(beta_hat, beta_hat_gmm)
end

@testset "OLS with starting value" begin
    N = 1_000
    beta = [0.25, 0.75]
    x1 = 10 .* randn(N)
    x2 = exp.(randn(N))
    X = [x1 x2]
    e = randn(N)

    y = (X * beta) .+ e
    beta_hat = (X' * X) \ (X' * y)

    model = OLS(y, X)
    _, beta_hat_gmm = solve(model, I, initial_theta = [0.3, 0.7])

    @test isapprox(beta_hat, beta_hat_gmm)
end


# We can also model OLS as a constrained GMM as follows:
# - Residual function r(ε, β) = ε
# - Constraint function c(ε, β) = ε - (y - Xβ)
# The Jacobians are as follows:
# ∂r/∂ε = I  ∂r/∂β = 0
# ∂c/∂ε = I  ∂c/∂β = X
struct OLSConstraints <: GMMModel
    y::Vector{Float64}
    X::Matrix{Float64}
end

function gmm_num_residuals(ols::OLSConstraints)
    return size(ols.X, 1)
end

function gmm_num_instruments(ols::OLSConstraints)
    return size(ols.X, 2)
end

function gmm_num_parameters(ols::OLSConstraints)
    return size(ols.X, 1) + size(ols.X, 2)
end

function gmm_num_constraints(ols::OLSConstraints)
    return size(ols.X, 1)
end

function gmm_instruments(ols::OLSConstraints)
    return ols.X
end

function gmm_residuals_constraints!(ols::OLSConstraints, theta, residuals, constraints)
    N, K = size(ols.X)
    epsilon, beta = theta[1:N], theta[N+1:end]
    @assert length(epsilon) == N
    @assert length(beta) == K

    residuals .= epsilon
    constraints .= epsilon .- ols.y .+ (ols.X * beta)
end

function gmm_residuals_constraints_jacobians!(ols::OLSConstraints, theta, residuals_jacobian, constraints_jacobian)
    N, K = size(ols.X)
    epsilon, beta = theta[1:N], theta[N+1:end]
    @assert length(epsilon) == N
    @assert length(beta) == K

    residuals_jacobian .= [I zeros(N, K)]
    constraints_jacobian .= [I ols.X]
end

@testset "OLS with constraints" begin
    N = 1_000
    beta = [0.75, 0.25]
    x1 = 10 .* randn(N)
    x2 = exp.(randn(N))
    X = [x1 x2]
    e = randn(N)

    y = (X * beta) .+ e
    beta_hat = (X' * X) \ (X' * y)

    model = OLSConstraints(y, X)
    _, theta_hat_gmm = solve(model, I)

    beta_hat_gmm = theta_hat_gmm[N+1:end]

    @test isapprox(beta_hat, beta_hat_gmm)
end


# We can also model OLS as a constrained GMM as follows:
# - Residual function r(ε, β) = ε
# - Constraint function c(ε, β) = [ε - (y - Xβ); X'(y - Xβ) ]
# The last constraint is redundant as it is the same as the moment condition.
# The Jacobians are as follows:
# ∂r/∂ε = I  ∂r/∂β = 0
# ∂c_1/∂ε = I  ∂c_1/∂β = X
# ∂c_2/∂ε = )  ∂c_2/∂β = -X'X
struct OLSRedundantConstraints <: GMMModel
    y::Vector{Float64}
    X::Matrix{Float64}
end

function gmm_num_residuals(ols::OLSRedundantConstraints)
    return size(ols.X, 1)
end

function gmm_num_instruments(ols::OLSRedundantConstraints)
    return size(ols.X, 2)
end

function gmm_num_parameters(ols::OLSRedundantConstraints)
    return size(ols.X, 1) + size(ols.X, 2)
end

function gmm_num_constraints(ols::OLSRedundantConstraints)
    return size(ols.X, 1) + size(ols.X, 2)
end

function gmm_instruments(ols::OLSRedundantConstraints)
    return ols.X
end

function gmm_residuals_constraints!(ols::OLSRedundantConstraints, theta, residuals, constraints)
    N, K = size(ols.X)
    epsilon, beta = theta[1:N], theta[N+1:end]
    @assert length(epsilon) == N
    @assert length(beta) == K

    residuals .= epsilon
    constraints[1:N] .= epsilon .- ols.y .+ (ols.X * beta)
    constraints[N+1:end] .= ols.X' * (ols.y - ols.X * beta)
end

function gmm_residuals_constraints_jacobians!(ols::OLSRedundantConstraints, theta, residuals_jacobian, constraints_jacobian)
    N, K = size(ols.X)
    epsilon, beta = theta[1:N], theta[N+1:end]
    @assert length(epsilon) == N
    @assert length(beta) == K

    residuals_jacobian .= [I zeros(N, K)]
    constraints_jacobian .= [ I ols.X ; zeros(K, N) (-ols.X' * ols.X) ]
end

@testset "OLS with redundant constraints" begin
    N = 1_000
    beta = [0.75, 0.25]
    x1 = 10 .* randn(N)
    x2 = exp.(randn(N))
    X = [x1 x2]
    e = randn(N)

    y = (X * beta) .+ e
    beta_hat = (X' * X) \ (X' * y)

    model = OLSRedundantConstraints(y, X)
    _, theta_hat_gmm = solve(model, I)

    beta_hat_gmm = theta_hat_gmm[N+1:end]

    @test isapprox(beta_hat, beta_hat_gmm)
end
