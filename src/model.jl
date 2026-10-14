export simulate!, negative_log_density

"Abstract base type for state space models that can assimilate observational data."
abstract type AbstractStateSpaceModel end

"""
    $(FUNCTIONNAME)(model, t)

Timestep `model` forward from observation time index `t` to `t + 1`.
Updates model state in place.
"""
function step! end

"""
    $(FUNCTIONNAME)(ȳ, model, t)

Compute mean of observed vector `y` for `model` at observation time index `t`.
Writes to `y` in place.
"""
function observe! end

"""
    $(FUNCTIONNAME)(model, v, t)

Update state of `model` using control variable vector `v` at time index `t`.
Updates model state in place.
"""
function update! end

"""
    $(FUNCTIONNAME)(model, t)

Compute observation covariance matrix for `model` at observation time index `t`.
This is the covariance of the Gaussian observation noise at time index `t` or
equivalently the covariance of the observation vector at time index `t` given
the model state at that time index.
"""
function observation_covariance end

"""
    $(FUNCTIONNAME)(model, t)

Compute control covariance matrix for `model` at time index `t`.
This is the covariance of the Gaussian prior distribution at time index `t`
on the control vector,
"""
function control_covariance end

"""
    $(FUNCTIONNAME)(model)

Get prognostic state of `model`.
"""
function state end

"""
    $(FUNCTIONNAME)(model, state)

Restore prognostic state of `model` to `state`, updating `model` in place.
"""
function restore_state! end

"""
    $(FUNCTIONNAME)(model)

Get numeric element type for `model` state, observation and control variable vectors.
"""
function eltype end

"""
    $(FUNCTIONNAME)(model)

Get observation vector dimension for `model`.
"""
function observation_dimension end

"""
    $(FUNCTIONNAME)(model)

Get control variable vector dimension for `model`.
"""
function control_dimension end

"""
Check state sequence `x_seq`, observation mean sequence `ȳ_seq` and control vector
sequence `v_seq` have consistent lengths and compute and return the length of the
subwindow length.
"""
function check_args_and_get_subwindow_length(x_seq, ȳ_seq, v_seq)
    @assert length(x_seq) == length(v_seq)
    subwindow_length, remainder = divrem(length(ȳ_seq), length(v_seq))
    @assert remainder == 0
    return subwindow_length
end

"""
Simulate a subwindow of observation times starting at index `t` for `model`, updating
its state in place using control variable vector `v` and simulating for as many
observation times as there are entries in `ȳ_subseq`, with the observation means
for each time index being written in place to `ȳ_subseq`.
"""
function simulate_subwindow!(model, ȳ_subseq::AbstractVector{Y}, v::V, t::Int) where {V,Y}
    update!(model, v, t)
    for s in 1:length(ȳ_subseq)
        observe!(ȳ_subseq[s], model, t + s - 1)
        step!(model, t + s - 1)
    end
    return nothing
end

"""
Simulate a sequence of states `x_seq` and observation mean vectors `ȳ_seq` for `model`
given a sequence of control variable vectors `v_seq`. The lengths of `x_seq` and `v_seq`
should match, with `x_seq[t]` the model state immediately _prior_ to updating with
control variable vector `v_seq[t]` at time index `t`. The length of `y_seq`
should be an integer multiple of the length of `v_seq` and `x_seq`, with this integer
multiple corresponding to the subwindow length between pairs of control variable updates
to the model state. States and observation means are written in place to `x_seq` and
`y_seq` and `model` is updated in place.
"""
function simulate!(x_seq, ȳ_seq, model, v_seq)
    subwindow_length = check_args_and_get_subwindow_length(x_seq, ȳ_seq, v_seq)
    for r in 1:length(x_seq)
        t = (r - 1) * subwindow_length + 1
        x_seq[r] = deepcopy(state(model))
        simulate_subwindow!(model, view(ȳ_seq, t:(t + subwindow_length - 1)), v_seq[r], t)
    end
    return nothing
end

"""
Compute the negative log density of the joint distribution on the sequence of observed
vectors `y_seq` and control variable vectors `v_seq` for `model` given a precomputed
sequence of observation mean vectors `ȳ_seq`.
"""
function negative_log_density(y_seq, ȳ_seq, v_seq, model)
    ℓ = sum((t, v) -> v' * control_covariance(model, t) \ v, enumerate(v_seq)) / 2
    for t in 1:length(y_seq)
        d = y_seq[t] - ȳ_seq[t]
        ℓ += d' * (observation_covariance(model, t) \ d) / 2
    end
    return ℓ
end
