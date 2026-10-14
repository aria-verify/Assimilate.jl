using Enzyme
using Reactant

struct Incremental4DVar{S} <: AbstractAssimilationMethod
    inner_solver::S
    inner_iterations::Int
    outer_iterations::Int
end

struct Direct4DVar{T} <: AbstractAssimilationMethod
    iterations::Int
    step_size::T
end

struct DiffWorkspace{V,X,Y,M}
    x::X
    δx_a::X
    δx_b::X
    ȳ_subseq::Vector{Y}
    δȳ_seq::Vector{Y}
    δv_seq::Vector{V}
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
    return objective(y_seq, ȳ_seq, v_seq, model)
end

function tangent_linear_model!(δȳ_seq, model, workspace, x_seq, v_seq, δv_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, δȳ_seq, v_seq)
    @trace for r in 1:length(x_seq)
        t = (r - 1) * subwindow_length + 1
        δȳ_subseq = view(δȳ_seq, t:(t + subwindow_length - 1))
        restore_state!(model, x_seq[r])
        Enzyme.make_zero!(workspace.dmodel)
        Enzyme.autodiff(
            Forward,
            simulate_subwindow!,
            Duplicated(model, workspace.dmodel),
            Duplicated(workspace.ȳ_subseq, δȳ_subseq),
            Duplicated(x_seq[r], workspace.δx_a),
            Duplicated(v_seq[r], δv_seq[r]),
            Const(t),
        )
    end
    return nothing
end

function adjoint!(δv_seq, model, workspace, x_seq, v_seq, δȳ_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, δȳ_seq, v_seq)
    fill_zero!(workspace.δx_a)
    δx_curr, δx_next = workspace.δx_a, workspace.δx_b
    for r in (length(x_seq) - 1):-1:1
        t = (r - 1) * subwindow_length + 1
        δȳ_subseq = view(δȳ_seq, t:(t + subwindow_length - 1))
        fill_zero!(δx_next)
        fill_zero!(δv_seq[r + 1])
        Enzyme.make_zero!(workspace.dmodel)
        Enzyme.autodiff(
            Reverse,
            simulate_subwindow!,
            Duplicated(workspace.x, δx_curr),
            Duplicated(model, workspace.dmodel),
            Duplicated(workspace.ȳ_subseq, δȳ_subseq),
            Duplicated(x_seq[r], δx_next),
            Duplicated(v_seq[r + 1], δv_seq[r + 1]),
            Const(t),
        )
        δx_curr, δx_next = δx_next, δx_curr
    end
    fill_zero!(δv_seq[1])
    Enzyme.autodiff(
        Reverse,
        initialize!,
        Duplicated(workspace.x_scratch, δx_curr),
        Const(model),
        Duplicated(v_seq[1], δv_seq[1]),
    )
    return nothing
end

function gauss_newton_product!(q_seq, δȳ_seq, model, workspace, x_seq, v_seq, p_seq)
    tangent_linear_model!(δȳ_seq, model, workspace, x_seq, v_seq, p_seq)
    for t in 1:length(δȳ_seq)
        δȳ_seq[t] .= observation_covar(model, t) \ δȳ_seq[t]
    end
    adjoint!(q_seq, model, workspace, x_seq, v_seq, δȳ_seq)
    for r in 1:length(v_seq)
        q_seq[r] .+= p_seq[r]
    end
    return nothing
end

struct CGWorkspace
    d_seq
    r_new_seq
    δȳ_seq
end

inner(v_a::Vector{T}, v_b::Vector{T}) where {T} = sum(((a, b),) -> a' * b, zip(v_a, v_b))

function solve_for_increment!(δv_seq, x_seq, ȳ_seq, y_seq, v_seq, model, method, workspace)
    for r in 1:length(δv_seq)
        δv_seq[r] .= 0.0
    end
    for t in 1:length(y_seq)
        d_seq[t] .= y_seq[t] .- ȳ_seq[t]
        d_seq[t] .= observation_noise_covar(model, t) \ d_seq[t]
    end
    adjoint!(r_seq, model, workspace, x_seq, v_seq, d_seq)
    for t in 1:length(y_seq)
        r_seq[t] .-= v_seq[t]
    end
    p_seq = copy(r_seq)
    for _ in 1:method.inner_iterations
        gauss_newton_product!(q_seq, δȳ_seq, model, workspace, x_seq, v_seq, p_seq)
        α = inner(r_seq, r_seq) / inner(r_seq, q_seq)
        δv_seq .+= α .* p_seq
        r_new_seq = r_seq .- α * q_seq
        β = inner(r_new_seq, r_new_seq) / inner(r_seq, r_seq)
        p_seq .*= β
        p_seq .+= r_new_seq
        r_seq .= r_new_seq
    end
end
