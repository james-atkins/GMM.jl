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
