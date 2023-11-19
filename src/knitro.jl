abstract type Cache end

using KNITRO: EvalResult
Base.@kwdef struct DenseCache <: Cache
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
        return get!(cache.residuals, thread_id) do
            Vector{Float64}(undef, cache.N)
        end
    finally
        unlock(cache.rlock)
    end
end


function get_residuals_jacobian!(cache::DenseCache, thread_id)
    lock(cache.jlock)
    try
        return get!(cache.residuals_jac, thread_id) do
            Matrix{Float64}(undef, cache.N, cache.K)
        end
    finally
        unlock(cache.jlock)
    end
end


Base.@kwdef struct SparseCache <: Cache
    N::Int  # Number of residuals
    K::Int  # Number of parameters
    M::Int  # Number of moment conditions
    C::Int  # Number of constraints

    r_jac_rows::Vector{Int64}
    r_jac_cols::Vector{Int64}

    mc_jac_rows::Vector{Int64}
    mc_jac_cols::Vector{Int64}

    rlock::Threads.SpinLock = Threads.SpinLock()
    residuals::Dict{Int32, Vector{Float64}} = Dict()  # N-vector to prevent reallocating storage for residuals

    jlock::Threads.SpinLock = Threads.SpinLock()
    r_jac::Dict{Int32, SparseArrays.FixedSparseCSC{Float64, Int64}} = Dict()  # N x K sparse matrix
    mc_jac::Dict{Int32, SparseArrays.FixedSparseCSC{Float64, Int64}} = Dict()  # (M+C) x K sparse matrix
end


function get_sparse_jacobians!(cache::SparseCache, thread_id)
    lock(cache.jlock)
    try
        r_jac = get!(cache.r_jac, thread_id) do
            SparseArrays.fixed(cache.r_jac_rows, cache.r_jac_cols, similar(cache.r_jac_rows, Float64), cache.N, cache.K)
        end

        mc_jac = get!(cache.mc_jac, thread_id) do
            SparseArrays.fixed(cache.mc_jac_rows, cache.mc_jac_cols, similar(cache.mc_jac_rows, Float64), cache.M + cache.C, cache.K)
        end

        return (r_jac, mc_jac)
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


function eval_constraints_dense_jac(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALGA
        @error "eval_constraints_dense_jac incorrectly called with eval type $(evalRequest.evalRequestCode)"
        return -1
    end

    model, cache = userParams

    theta = @view evalRequest.x[cache.M+1:end]
    r_jac = get_residuals_jacobian!(cache, evalRequest.threadID)

    jac = reshape(evalResult.jac, cache.M + cache.C, cache.K)
    m_jac = @view jac[1:cache.M, :]
    c_jac = @view jac[cache.M+1:end, :]

    # Compute the residual and the constraint Jacobians in-place
    gmm_residuals_constraints_jacobians!(model, theta, r_jac, c_jac)
    mul!(m_jac, gmm_instruments(model)', r_jac, -1, 0)

    return 0
end


function eval_constraints_sparse_jac(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALGA
        @error "eval_constraints_sparse_jac incorrectly called with eval type $(evalRequest.evalRequestCode)"
        return -1
    end

    model, cache = userParams

    theta = @view evalRequest.x[cache.M+1:end]

    r_jac, mc_jac = get_sparse_jacobians!(cache, evalRequest.threadID)
    m_jac = @view mc_jac[1:cache.M, :]
    c_jac = @view mc_jac[cache.M+1:end, :]

    # Compute the residual and the constraint Jacobians in-place
    gmm_residuals_constraints_jacobians!(model, theta, r_jac, c_jac)
    # mul!(m_jac, gmm_instruments(model)', r_jac, -1, 0)
    m_jac .= -gmm_instruments(model)' * r_jac

    _, _, nzval = findnz(mc_jac)
    evalResult.jac[:] = nzval

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

    try
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
        sparsity = gmm_sparsity(model)
        if sparsity === nothing
            # Assume that the Jacobian is dense
            KNITRO.KN_set_cb_grad(
                kc,
                cb_con,
                eval_constraints_dense_jac,
                nV = 0,  # Does not evaluate the gradient of the objective
                # Column order major
                jacIndexCons = repeat(con_indices, outer = length(var_theta_indices)),
                jacIndexVars = repeat(var_theta_indices, inner = length(con_indices))
            )

            cache = DenseCache(N = N, K = K, M = M, C = C)
        else
            r_jac_sparsity, c_jac_sparsity = sparsity
            r_jac_rows, r_jac_cols, _ = findnz(r_jac_sparsity)

            if size(r_jac_sparsity, 1) != N || size(r_jac_sparsity, 2) != K
                throw(DimensionMismatch("residual jacobian sparsity has wrong size! want $((N, K)) but got $(size(r_jac_sparsity))"))
            end

            if size(c_jac_sparsity, 1) != C || size(c_jac_sparsity, 2) != K
                throw(DimensionMismatch("constraint jacobian sparsity has wrong size! want $((C, K)) but got $(size(c_jac_sparsity))"))
            end

            # We need to compute the sparsity of ∂r/∂θ

            # Calculate the sparsity of -Z'∂r/∂θ
            Z = gmm_instruments(model)
            Z_sparsity = @. Z != 0
            m_jac_sparsity = -Z_sparsity' * r_jac_sparsity

            mc_jac_sparsity = sparse_vcat(m_jac_sparsity, c_jac_sparsity)
            mc_jac_rows, mc_jac_cols, _ = findnz(mc_jac_sparsity)

            # Minus one to convert from Julia to Knitro indexing
            jac_index_cons = mc_jac_rows  .- 1
            jac_index_vars = @. mc_jac_cols + M - 1 

            KNITRO.KN_set_cb_grad(
                kc,
                cb_con,
                eval_constraints_sparse_jac,
                nV = 0,  # Does not evaluate the gradient of the objective
                jacIndexCons = jac_index_cons,
                jacIndexVars = jac_index_vars
            )

            cache = SparseCache(
                N = N,
                K = K,
                M = M,
                C = C, 
                r_jac_rows = r_jac_rows,
                r_jac_cols = r_jac_cols,
                mc_jac_rows = mc_jac_rows,
                mc_jac_cols = mc_jac_cols
            )
        end

        KNITRO.KN_set_cb_user_params(kc, cb_con, (model, cache))

        status = KNITRO.KN_solve(kc)
        if status != KNITRO.KN_RC_OPTIMAL_OR_SATISFACTORY
            error("Knitro could not find locally optimal solution!")
        end

        theta_opt = KNITRO.KN_get_var_primal_values(kc, var_theta_indices)
        obj_value = KNITRO.KN_get_obj_value(kc)

        return obj_value, theta_opt
    finally
        KNITRO.KN_free(kc)
    end
end
