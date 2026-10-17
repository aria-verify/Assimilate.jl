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
    r_new_seq::Vector{V}
end

function assimilate!(method::Incremental4DVar, model, v_seq, y_seq)
    (; x_seq, ȳ_seq, dv_seq) = method
    for _ in 1:method.outer_iterations
        simulate!(x_seq, ȳ_seq, model, v_seq)
        solve_for_increment!(method, dv_seq, model, v_seq, x_seq, ȳ_seq, y_seq)
        v_seq .+= dv_seq
    end
    simulate!(x_seq, ȳ_seq, model, v_seq)
    return negative_log_density(y_seq, ȳ_seq, v_seq, model)
end

function tangent_linear_model!(ȳ_seq, dȳ_seq, model, dmodel, v_seq, dv_seq, x_seq)
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
            Duplicated(
                view(ȳ_seq, t:(t + subwindow_length - 1)),
                view(dȳ_seq, t:(t + subwindow_length - 1)),
            ),
            Duplicated(v_seq[r], dv_seq[r]),
            Const(t),
        )
    end
    return nothing
end

function adjoint!(ȳ_seq, dv_seq, model, dmodel, v_seq, x_seq, dȳ_seq)
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
            Duplicated(
                view(ȳ_seq, t:(t + subwindow_length - 1)),
                view(dȳ_seq, t:(t + subwindow_length - 1)),
            ),
            Duplicated(v_seq[r], dv_seq[r]),
            Const(t),
        )
    end
    return nothing
end

function gauss_newton_product!(q_seq, ȳ_seq, dȳ_seq, model, dmodel, v_seq, x_seq, p_seq)
    tangent_linear_model!(ȳ_seq, dȳ_seq, model, dmodel, v_seq, p_seq, x_seq)
    for t in 1:length(dȳ_seq)
        dȳ_seq[t] .= observation_covariance(model, t) \ dȳ_seq[t]
    end
    adjoint!(ȳ_seq, q_seq, model, dmodel, v_seq, x_seq, dȳ_seq)
    for r in 1:length(v_seq)
        q_seq[r] .+= control_covariance(model, r) \ p_seq[r]
    end
    return nothing
end

inner(v_a::Vector{T}, v_b::Vector{T}) where {T} = sum(((a, b),) -> a' * b, zip(v_a, v_b))
axpy_seq!(y, a, x) = foreach((yi, xi) -> (yi .+= a .* xi), y, x)

function solve_for_increment!(
    method::Incremental4DVar, dv_seq, model, v_seq, x_seq, ȳ_seq, y_seq
)
    (; p_seq, q_seq, r_seq, r_new_seq, dmodel, dȳ_seq) = method
    for r in 1:length(dv_seq)
        dv_seq[r] .= 0.0
    end
    for t in 1:length(y_seq)
        dȳ_seq[t] .= y_seq[t] .- ȳ_seq[t]
        dȳ_seq[t] .= observation_covariance(model, t) \ dȳ_seq[t]
    end
    for r in 1:length(v_seq)
        r_seq[r] .= -control_covariance(model, r) \ v_seq[r]
    end
    adjoint!(ȳ_seq, r_seq, model, dmodel, v_seq, x_seq, dȳ_seq)
    foreach(copyto!, p_seq, r_seq)
    rr = inner(r_seq, r_seq)
    for _ in 1:method.inner_iterations
        gauss_newton_product!(q_seq, ȳ_seq, dȳ_seq, model, dmodel, v_seq, x_seq, p_seq)
        α = rr / inner(p_seq, q_seq)
        axpy_seq!(dv_seq, α, p_seq)
        axpy_seq!(r_seq, -α, q_seq)
        rr_new = inner(r_seq, r_seq)
        sqrt(rr_new) < method.inner_tolerance && break
        foreach((p, r) -> (p .= r .+ (rr_new / rr) .* p), p_seq, r_seq)
        rr = rr_new
    end
end
