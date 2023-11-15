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
