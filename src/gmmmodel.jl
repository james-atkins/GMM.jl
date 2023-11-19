abstract type GMMModel end


function gmm_num_instruments(model::GMMModel)
    # M instruments
    error("gmm_num_instruments not defined for model type $(typeof(model))")
end


function gmm_num_parameters(model::GMMModel)
    # K instruments
    error("gmm_num_parameters not defined for model type $(typeof(model))")
end


function gmm_num_residuals(model::GMMModel)
    # N observations
    error("gmm_num_residuals not defined for model type $(typeof(model))")
end


function gmm_num_constraints(model::GMMModel)
    # C observations
    return 0
end


function gmm_instruments(model::GMMModel)
    # Returns a NxM matrix
    error("gmm_instruments not defined for model type $(typeof(model))")
end


function gmm_residuals_constraints!(model::GMMModel, theta, residuals, constraints)
    error("gmm_residuals_constraints! not defined for model type $(typeof(model))")
end


function gmm_residuals_constraints_jacobians!(model::GMMModel, theta, residuals_jacobian, constraints_jacobian)
    error("gmm_residuals_constraints_jacobians! not defined for model type $(typeof(model))")
end


function gmm_sparsity(model::GMMModel)
    return nothing
end

