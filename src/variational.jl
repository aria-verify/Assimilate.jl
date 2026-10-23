using Enzyme

struct Incremental4DVar{T,V,X,Y,M} <: AbstractAssimilationMethod
    inner_iterations::Int
    outer_iterations::Int
    inner_tolerance::T
    x_seq::Vector{X}
    ȳ_seq::Vector{Y}
    dv_seq::Vector{V}
    dȳ_seq::Vector{Y}
    dmodel::M
    p_seq::Vector{V}
    q_seq::Vector{V}
    r_seq::Vector{V}
end

function assimilate!(method::Incremental4DVar, model, v_seq, y_seq)
    (; x_seq, ȳ_seq, dv_seq) = method
    for _ in 1:method.outer_iterations
        simulate!(model, v_seq, x_seq, ȳ_seq, model)
        solve_for_increment!(method, dv_seq, model, v_seq, x_seq, ȳ_seq, y_seq)
        v_seq .-= dv_seq
    end
    simulate!(model, v_seq, x_seq, ȳ_seq)
    return negative_log_density(model, v_seq, y_seq, ȳ_seq, v_seq)
end

function tlm_simulate!(model, dmodel, v_seq, dv_seq, x_seq, ȳ_seq, dȳ_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, dȳ_seq, v_seq)
    Enzyme.remake_zero!(dmodel)
    restore_state!(model, x_seq[1])
    for r in 1:length(x_seq)
        t = (r - 1) * subwindow_length + 1
        Enzyme.autodiff(
            Forward,
            simulate_subwindow!,
            Const,
            Duplicated(model, dmodel),
            Duplicated(v_seq[r], dv_seq[r]),
            Duplicated(
                view(ȳ_seq, t:(t + subwindow_length - 1)),
                view(dȳ_seq, t:(t + subwindow_length - 1)),
            ),
            Const(t),
        )
    end
    return nothing
end

function adjoint_simulate!(model, dmodel, v_seq, dv_seq, x_seq, ȳ_seq, dȳ_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, dȳ_seq, v_seq)
    Enzyme.remake_zero!(dmodel)
    for r in length(x_seq):-1:1
        restore_state!(model, x_seq[r])
        t = (r - 1) * subwindow_length + 1
        Enzyme.autodiff(
            Reverse,
            simulate_subwindow!,
            Const,
            Duplicated(model, dmodel),
            Duplicated(v_seq[r], dv_seq[r]),
            Duplicated(
                view(ȳ_seq, t:(t + subwindow_length - 1)),
                view(dȳ_seq, t:(t + subwindow_length - 1)),
            ),
            Const(t),
        )
    end
    return nothing
end

function gauss_newton_product!(
    model, dmodel, v_seq, dv_seq_in, dv_seq_out, x_seq, ȳ_seq, dȳ_seq
)
    tlm_simulate!(model, dmodel, v_seq, dv_seq_in, x_seq, ȳ_seq, dȳ_seq)
    for t in 1:length(dȳ_seq)
        dȳ_seq[t] .= observation_covariance(model, t) \ dȳ_seq[t]
    end
    adjoint_simulate!(model, dmodel, v_seq, dv_seq_out, x_seq, ȳ_seq, dȳ_seq)
    for r in 1:length(dv_seq_out)
        dv_seq_out[r] .+= control_covariance(model, r) \ dv_seq_in[r]
    end
    return nothing
end

function grad_negative_log_density!(model, dmodel, v_seq, dv_seq, x_seq, ȳ_seq, dȳ_seq, y_seq)
    Enzyme.remake_zero!(dmodel)
    zero_seq!(dv_seq)
    for t in 1:length(y_seq)
        dȳ_seq[t] .= ȳ_seq[t] .- y_seq[t]
        dȳ_seq[t] .= observation_covariance(model, t) \ dȳ_seq[t]
    end
    adjoint_simulate!(model, dmodel, v_seq, dv_seq, x_seq, ȳ_seq, dȳ_seq)
    for r in 1:length(v_seq)
        dv_seq[r] .+= control_covariance(model, r) \ v_seq[r]
    end
end

inner(v_a::Vector{T}, v_b::Vector{T}) where {T} = sum(((a, b),) -> a' * b, zip(v_a, v_b))
zero_seq!(x) = foreach(xi -> xi .= 0.0, x)
axpy_seq!(y, a, x) = foreach((yi, xi) -> (yi .+= a .* xi), y, x)

function solve_for_increment!(
    method::Incremental4DVar, dv_seq, model, v_seq, x_seq, ȳ_seq, y_seq
)
    (; p_seq, q_seq, r_seq, dmodel, dȳ_seq) = method
    zero_seq!(dv_seq)
    grad_negative_log_density!(model, dmodel, v_seq, r_seq, x_seq, ȳ_seq, dȳ_seq, y_seq)
    foreach(copyto!, p_seq, r_seq)
    rr = inner(r_seq, r_seq)
    for _ in 1:method.inner_iterations
        gauss_newton_product!(model, dmodel, v_seq, p_seq, q_seq, x_seq, ȳ_seq, dȳ_seq)
        α = rr / inner(p_seq, q_seq)
        axpy_seq!(dv_seq, α, p_seq)
        axpy_seq!(r_seq, -α, q_seq)
        rr_new = inner(r_seq, r_seq)
        sqrt(rr_new) < method.inner_tolerance && break
        foreach((p, r) -> (p .= r .+ (rr_new / rr) .* p), p_seq, r_seq)
        rr = rr_new
    end
end
