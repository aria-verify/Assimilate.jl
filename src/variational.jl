using Enzyme

struct Incremental4DVar{S} <: AbstractAssimilationMethod
    inner_solver::S
    inner_iterations::Int
    outer_iterations::Int
end

struct DiffWorkspace{Y,M}
    ȳ_subseq::Vector{Y}
    dmodel::M
end

function assimilate!(
    x_seq, ȳ_seq, v_seq, δv_seq, model, workspace, y_seq, method::Incremental4DVar
)
    for _ in 1:method.outer_iterations
        simulate!(x_seq, ȳ_seq, v_seq, model)
        solve_for_increment!(δv_seq, x_seq, ȳ_seq, y_seq, v_seq, model, method, workspace)
        v_seq .+= δv_seq
    end
    simulate!(x_seq, ȳ_seq, model, v_seq)
    return negative_log_density(y_seq, ȳ_seq, v_seq, model)
end

function tangent_linear_model!(δȳ_seq, model, workspace, x_seq, v_seq, δv_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, δȳ_seq, v_seq)
    Enzyme.remake_zero!(workspace.dmodel)
    restore_state!(model, x_seq[1])
    for r in 1:length(x_seq)
        t = (r - 1) * subwindow_length + 1
        Enzyme.autodiff(
            Forward,
            simulate_subwindow!,
            Const,
            Duplicated(model, workspace.dmodel),
            Duplicated(workspace.ȳ_subseq, δȳ_seq[t:(t + subwindow_length - 1)]),
            Duplicated(v_seq[r], δv_seq[r]),
            Const(t),
        )
    end
    return nothing
end

function adjoint!(δv_seq, model, workspace, x_seq, v_seq, δȳ_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, δȳ_seq, v_seq)
    Enzyme.remake_zero!(workspace.dmodel)
    for r in length(x_seq):-1:1
        restore_state!(model, x_seq[r])
        t = (r - 1) * subwindow_length + 1
        Enzyme.make_zero!(δv_seq[r])
        Enzyme.autodiff(
            Reverse,
            simulate_subwindow!,
            Const,
            Duplicated(model, workspace.dmodel),
            Duplicated(workspace.ȳ_subseq, δȳ_seq[t:(t + subwindow_length - 1)]),
            Duplicated(v_seq[r], δv_seq[r]),
            Const(t),
        )
    end
    return nothing
end

function gauss_newton_product!(q_seq, δȳ_seq, model, workspace, x_seq, v_seq, p_seq)
    tangent_linear_model!(δȳ_seq, model, workspace, x_seq, v_seq, p_seq)
    for t in 1:length(δȳ_seq)
        δȳ_seq[t] .= observation_covariance(model, t) \ δȳ_seq[t]
    end
    adjoint!(q_seq, model, workspace, x_seq, v_seq, δȳ_seq)
    for r in 1:length(v_seq)
        q_seq[r] .+= control_covariance(model, r) \ p_seq[r]
    end
    return nothing
end

struct CGWorkspace
    d_seq
    p_seq
    q_seq
    r_seq
    r_new_seq
    δȳ_seq
end

inner(v_a::Vector{T}, v_b::Vector{T}) where {T} = sum(((a, b),) -> a' * b, zip(v_a, v_b))

function solve_for_increment!(
    δv_seq, x_seq, ȳ_seq, y_seq, v_seq, model, method, diff_workspace, cg_workspace
)
    for r in 1:length(δv_seq)
        δv_seq[r] .= 0.0
    end
    for t in 1:length(y_seq)
        cg_workspace.d_seq[t] .= y_seq[t] .- ȳ_seq[t]
        cg_workspace.d_seq[t] .= observation_covariance(model, t) \ cg_workspace.d_seq[t]
    end
    adjoint!(r_seq, model, diff_workspace, x_seq, v_seq, d_seq)
    for t in 1:length(y_seq)
        cg_workspace.r_seq[t] .-= v_seq[t]
    end
    cg_workspace.p_seq .= r_seq
    for _ in 1:method.inner_iterations
        gauss_newton_product!(
            cg_workspace.q_seq,
            δȳ_seq,
            model,
            diff_workspace,
            x_seq,
            v_seq,
            cg_workspace.p_seq,
        )
        α =
            inner(cg_workspace.r_seq, cg_workspace.r_seq) /
            inner(cg_workspace.r_seq, cg_workspace.q_seq)
        δv_seq .+= α .* cg_workspace.p_seq
        cg_workspace.r_new_seq .= cg_workspace.r_seq .- α * cg_workspace.q_seq
        β =
            inner(cg_workspace.r_new_seq, cg_workspace.r_new_seq) /
            inner(cg_workspace.r_seq, cg_workspace.r_seq)
        cg_workspace.p_seq .*= β
        cg_workspace.p_seq .+= cg_workspace.r_new_seq
        cg_workspace.r_seq .= cg_workspace.r_new_seq
    end
end
