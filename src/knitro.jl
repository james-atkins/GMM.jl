

Base.@kwdef struct Cache
    N::Int  # Number of residuals
    K::Int  # Number of parameters
    M::Int  # Number of moment conditions
    C::Int  # Number of constraints

    rlock::Threads.SpinLock = Threads.SpinLock()
    residuals::Dict{Int32, Vector{Float64}} = Dict()  # N-vector to prevent reallocating storage for residuals

    jlock::Threads.SpinLock = Threads.SpinLock()
    residuals_jac::Dict{Int32, Matrix{Float64}} = Dict()  # N x K matrix to prevent reallocating storage
end

function get_residuals!(cache::Cache, thread_id)
    lock(cache.rlock)
    try
        return get!(cache.residuals, thread_id, Vector{Float64}(undef, cache.N))
    finally
        unlock(cache.rlock)
    end
end


function get_residuals_jacobian!(cache::Cache, thread_id)
    lock(cache.jlock)
    try
        return get!(cache.residuals_jac, thread_id, Matrix{Float64}(undef, cache.N, cache.K))
    finally
        unlock(cache.jlock)
    end
end


function eval_constraints(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALFC
        @error "eval_constraints incorrectly called with eval type $(evalRequest.evalRequestCode)"
        return -1
    end

    model, cache = userParams

    theta = @view evalRequest.x[cache.M+1:end]
    r = get_residuals!(cache, evalRequest.threadID)

    m = @view evalResult.c[1:cache.M]
    c = @view evalResult.c[cache.M+1:end]

    # Compute the residuals and the constraints in-place
    gmm_residuals_constraints!(model, theta, r, c)
    mul!(m, gmm_instruments(model)', r, -1, 0)

    return 0
end


function eval_constraints_jac(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALGA
        @error "eval_constraints_jac incorrectly called with eval type $(evalRequest.evalRequestCode)"
        return -1
    end

    model, cache = userParams

    theta = @view evalRequest.x[cache.M+1:end]
    r_jac = get_residuals_jacobian!(cache, evalRequest.threadID)

    jac = reshape(evalResult.jac, cache.K, cache.M + cache.C)
    m_jac = @view jac[1:cache.M, :]
    c_jac = @view jac[cache.M+1:end, :]

    # Compute the residual and the constraint Jacobians in-place
    gmm_residuals_constraints_jacobians!(model, theta, r_jac, c_jac)
    mul!(m_jac, gmm_instruments(model)', r_jac, -1, 0)

    return 0
end


function solve(model::GMMModel, W)
    Base.require_one_based_indexing(W)

    N = gmm_num_residuals(model)
    K = gmm_num_parameters(model)
    M = gmm_num_instruments(model)
    C = gmm_num_constraints(model)

    if !isa(W, UniformScaling) && size(W, 1) != M
        throw(DimensionMismatch("W has incompatible size"))
    end

    if !issymmetric(W)
        throw(ArgumentError("W should be symmetric"))
    end

    # min_{m, θ} m' W m
    # s.t.
    # * m - Z' r(θ) = 0
    # *        c(θ) = 0
    #
    # m is M x 1
    # θ is K x 1
    # c(θ) is C x 1

    kc = KNITRO.KN_new()

    # Add the variables
    var_m_indices = KNITRO.KN_add_vars(kc, M)
    var_theta_indices = KNITRO.KN_add_vars(kc, K)

    #######################################
    #        Add the GMM objective        #
    #######################################

    # In matrix algebra, the GMM objective is m' W m.
    # We have M own-quadratic terms plus M(M-1)/2 cross-quadratic terms. E.g. if M=4, we have the following:
    # a² b² c² d²
    # ab ac ad
    # bc bd
    # cd
    num_obj_terms = M + div(M * (M - 1), 2)

    coefs = Vector{Float64}(undef, 0)
    sizehint!(coefs, num_obj_terms)

    index_vars_1 = Vector{eltype(var_m_indices)}(undef, 0)
    sizehint!(index_vars_1, num_obj_terms)

    index_vars_2 = Vector{eltype(var_m_indices)}(undef, 0)
    sizehint!(index_vars_2, num_obj_terms)

    # This presumes that W is column-major order.
    # TODO: make this more general and don't require one based indexing
    for i = 1:M
        for j = i:M
            if i == j
                push!(coefs, W[j, i])
            else
                push!(coefs, 2 * W[j, i])
            end
            push!(index_vars_1, var_m_indices[i])
            push!(index_vars_2, var_m_indices[j])
        end
    end

    KNITRO.KN_add_obj_quadratic_struct(kc, index_vars_1, index_vars_2, coefs)

    #######################################
    #        Add the constraints          #
    #######################################

    # The constrains are
    #   m - Z' r(θ) = 0  (M constraints)
    #   c(θ)        = 0  (C constraints)

    con_indices = KNITRO.KN_add_cons(kc, M + C)
    con_m_indices = con_indices[1:M]
    KNITRO.KN_set_con_eqbnds_all(kc, fill(0.0, M + C))

    # Add the linear constraints for m
    KNITRO.KN_add_con_linear_struct(kc, con_m_indices, var_m_indices, fill(1.0, M))

    # Add the non-linear, -Z'r(θ) and c(θ), part of the constraints. As r(θ) and c(θ) may contain shared
    # calculations, add a single callback that calculates them both at the same time.
    cb_con = KNITRO.KN_add_eval_callback(kc, false, con_indices, eval_constraints)

    # Compute the gradient of the non-linear part of the constraints, [-Z'∂r/∂θ ∂C/∂θ]'
    # As ∂C/∂m = 0, we only need to compute the Jacobian wrt θ
    KNITRO.KN_set_cb_grad(
        kc,
        cb_con,
        eval_constraints_jac,
        nV = 0,  # Does not evaluate the gradient of the objective
        # Column order major
        jacIndexCons = repeat(con_indices, outer = length(var_theta_indices)),
        jacIndexVars = repeat(var_theta_indices, inner = length(con_indices)),
    )

    cache = Cache(N = N, K = K, M = M, C = C)
    KNITRO.KN_set_cb_user_params(kc, cb_con, (model, cache))

    nStatus = KNITRO.KN_solve(kc)
    nStatus, objSol, x, _ = KNITRO.KN_get_solution(kc)

    println()
    println("Knitro converged with final status = ", nStatus)
    println("  optimal objective value  = ", objSol)
    println("  optimal primal values x  = ", x)
    println("  feasibility violation    = ", KNITRO.KN_get_abs_feas_error(kc))
    println("  KKT optimality violation = ", KNITRO.KN_get_abs_opt_error(kc))

    KNITRO.KN_free(kc)
end