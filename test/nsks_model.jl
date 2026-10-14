"""
Non-stationary Kuramoto-Sivashinksy state space model.

Adapted from Kuramoto-Sivashinksy model example in Oceananigans model interface documentation

https://github.com/CliMA/Oceananigans.jl/blob/fec351f0/docs/src/developer_docs/model_interface.md
"""

using LinearAlgebra

using Oceananigans
using Oceananigans.Models: AbstractModel
using Oceananigans.TimeSteppers: Clock, tick!, tick_stage!
using Oceananigans: TendencyCallsite, UpdateStateCallsite
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TimeSteppers: maybe_prepare_first_time_step!, time_step!, update_state!
import Oceananigans.Fields: set!
import Oceananigans.Architectures: architecture

using Assimilate
using RandomFields

@kwdef struct NSKSModelParameters{T}
    κ₀::T = 1.0
    κ₁::T = 0.0
    γ::T = Inf
end

function Base.summary(parameters::NSKSModelParameters)
    return "(κ₀=$(parameters.κ₀), κ₁=$(parameters.κ₁), γ=$(parameters.γ))"
end

"""Non-stationary Kuramoto-Sivashinksy model"""
struct NSKSModel{P,G,C,U,T} <: AbstractModel{Nothing,Nothing}
    parameters::P
    grid::G
    clock::C
    solution::U
    tendencies::T
end

κ(t, p::NSKSModelParameters) = p.κ₀ + p.κ₁ * exp(t / p.γ)

function NSKSModel(parameters, grid)
    size(grid, 2) == 1 && size(grid, 3) == 1 ||
        throw(ArgumentError("KuramotoSivashinskyModel requires a 1D grid in x"))
    clock = Clock{eltype(grid)}(; time=zero(eltype(grid)))
    solution = CenterField(grid)
    tendencies = (Gⁿ=CenterField(grid), G⁻=CenterField(grid))
    return NSKSModel(parameters, grid, clock, solution, tendencies)
end

function NSKSModel(parameters; grid_size=256, domain_size=32π, halo_size=4)
    grid = RectilinearGrid(;
        size=grid_size, x=(0, domain_size), topology=(Periodic, Flat, Flat), halo=halo_size
    )
    return NSKSModel(parameters, grid)
end

function Base.summary(model::NSKSModel)
    grid_str = summary(model.grid)
    param_str = summary(model.parameters)
    return "NSKSModel on $grid_str with parameters $param_str"
end

architecture(model::NSKSModel) = model.grid.architecture
Base.eltype(model::NSKSModel) = eltype(model.grid)

"""Compute the right-hand side of the NSKS equation: -∂²u - κ(t) * ∂⁴u - ½∂ₓ(u²)"""
function compute_tendencies!(model::NSKSModel)
    u = model.solution
    t = model.clock.time
    Gⁿ = model.tendencies.Gⁿ
    ∂²u = ∂x(∂x(u))
    ∂⁴u = ∂x(∂x(∂²u))
    ∂u² = @at (Center, Center, Center) ∂x(u^2) / 2
    Gⁿ .= -∂²u .- κ(t, model.parameters) * ∂⁴u .- ∂u²
    return nothing
end

function update_state!(model::NSKSModel, callbacks=[]; compute_tendencies=true)
    fill_halo_regions!(model.solution)
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]
    compute_tendencies && compute_tendencies!(model)
    return nothing
end

function time_step!(model::NSKSModel, Δt; callbacks=[])
    # First stage: initialize
    maybe_prepare_first_time_step!(model, Δt, callbacks)
    [callback(model) for callback in callbacks if callback.callsite isa TendencyCallsite]

    # RK3 coefficients (Williamson's low-storage scheme)
    FT = eltype(model)
    γ¹, γ², γ³ = FT(8/15), FT(5/12), FT(3/4)
    ζ², ζ³ = -FT(17/60), -FT(5/12)

    u = parent(model.solution)
    Gⁿ = parent(model.tendencies.Gⁿ)
    G⁻ = parent(model.tendencies.G⁻)

    # Stage 1: u = u + Δt * γ¹ * Gⁿ
    u .+= Δt * γ¹ .* Gⁿ
    G⁻ .= Gⁿ
    tick_stage!(model.clock, Δt * γ¹)
    update_state!(model, callbacks)

    # Stage 2: u = u + Δt * (γ² * Gⁿ + ζ² * G⁻)
    u .+= Δt * γ² .* Gⁿ .+ Δt * ζ² .* G⁻
    G⁻ .= Gⁿ
    tick_stage!(model.clock, Δt * (γ² + ζ²))
    update_state!(model, callbacks)

    # Stage 3: u = u + Δt * (γ³ * Gⁿ + ζ³ * G⁻)
    u .+= Δt * γ³ .* Gⁿ .+ Δt * ζ³ .* G⁻
    tick_stage!(model.clock, Δt * (γ³ + ζ³), Δt)  # final stage: pass full Δt as step size
    update_state!(model, callbacks)

    return nothing
end

struct NSKSStateSpaceModel{M,T,P,F,GI,GE} <: Assimilate.AbstractStateSpaceModel
    underlying_model::M
    n_timestep_per_obs::Int
    timestep::T
    observation_points::Vector{P}
    observation_noise_std::T
    model_error_buffer::F
    initial_state_generator::GI
    model_error_generator::GE
end

function NSKSStateSpaceModel(
    model::NSKSModel,
    n_time_step_per_obs,
    timestep,
    observation_points,
    observation_noise_std,
    initial_state_parameters,
    model_error_parameters,
)
    solution_field = model.solution
    model_error_buffer = similar(solution_field)
    initial_state_generator = RandomFieldGenerator(solution_field, initial_state_parameters)
    model_error_generator = RandomFieldGenerator(solution_field, model_error_parameters)
    return NSKSStateSpaceModel(
        model,
        n_time_step_per_obs,
        timestep,
        observation_points,
        observation_noise_std,
        model_error_buffer,
        initial_state_generator,
        model_error_generator,
    )
end

function NSKSStateSpaceModel(
    model_parameters::NSKSModelParameters,
    grid,
    n_time_step_per_obs,
    timestep,
    observation_points,
    observation_noise_std,
    initial_state_parameters,
    model_error_parameters,
)
    model = NSKSModel(model_parameters, grid)
    return NSKSStateSpaceModel(
        model,
        n_time_step_per_obs,
        timestep,
        observation_points,
        observation_noise_std,
        initial_state_parameters,
        model_error_parameters,
    )
end

function Assimilate.state(model::NSKSStateSpaceModel)
    return (; solution=model.underlying_model.solution, clock=model.underlying_model.clock)
end

function Assimilate.restore_state!(model::NSKSStateSpaceModel, state)
    model.underlying_model.solution ≢ state.solution &&
        set!(model.underlying_model.solution, state.solution)
    model.underlying_model.clock ≢ state.clock &&
        set!(model.underlying_model.clock, state.clock)
    return nothing
end

function Assimilate.step!(model::NSKSStateSpaceModel, t)
    for _ in 1:model.n_timestep_per_obs
        time_step!(model.underlying_model, model.timestep)
    end
    return nothing
end

function Assimilate.observation_dimension(model::NSKSStateSpaceModel, t)
    return length(model.observation_points)
end

function Assimilate.control_dimension(model::NSKSStateSpaceModel, t)
    return length(model.underlying_model.solution)
end

function Assimilate.update!(model::NSKSStateSpaceModel, v, t)
    field = model.underlying_model.solution
    error = model.model_error_buffer
    if t == 1
        generate!(field, model.initial_state_generator, v)
    else
        generate!(error, model.model_error_generator, v)
        set!(field, field + error)
    end
    return nothing
end

function Assimilate.observe!(y, model::NSKSStateSpaceModel, t)
    for (i, point) in enumerate(model.observation_points)
        y[i] = Oceananigans.interpolate((point,), model.underlying_model.solution)
    end
    return nothing
end

function Assimilate.observation_covariance(model::NSKSStateSpaceModel{M,T}, t) where {M,T}
    return UniformScaling{T}(model.observation_noise_std^2)
end

function Assimilate.control_covariance(model::NSKSStateSpaceModel{M,T}, t) where {M,T}
    return UniformScaling{T}(model.observation_noise_std^2)
end
