using LinearAlgebra
using PDMats

using Assimilate

struct LinearGaussianModel{
    T,
    S<:AbstractMatrix{T},
    O<:AbstractMatrix{T},
    M<:AbstractVector{T},
    I<:AbstractPDMat{T},
    N<:AbstractPDMat{T},
    R<:AbstractPDMat{T},
} <: Assimilate.AbstractStateSpaceModel{T}
    state::Vector{T}
    state_buffer::Vector{T}
    state_transition_matrix::S
    observation_matrix::O
    initial_state_mean::M
    initial_state_covariance::I
    state_noise_covariance::N
    observation_noise_covariance::R
end

function LinearGaussianModel(
    state_transition_matrix::AbstractMatrix{T},
    observation_matrix::AbstractMatrix{T},
    initial_state_mean::AbstractVector{T},
    initial_state_covariance::AbstractMatrix{T},
    state_noise_covariance::AbstractMatrix{T},
    observation_noise_covariance::AbstractMatrix{T},
) where {T}
    state_dimension = size(state_transition_matrix, 1)
    @assert size(state_transition_matrix, 2) == state_dimension
    @assert size(observation_matrix, 2) == state_dimension
    @assert length(initial_state_mean) == state_dimension
    @assert size(initial_state_covariance) == (state_dimension, state_dimension)
    @assert size(state_noise_covariance) == (state_dimension, state_dimension)
    state = Vector{T}(undef, state_dimension)
    state_buffer = Vector{T}(undef, state_dimension)
    return LinearGaussianModel(
        state,
        state_buffer,
        state_transition_matrix,
        observation_matrix,
        initial_state_mean,
        initial_state_covariance,
        state_noise_covariance,
        observation_noise_covariance,
    )
end

function stochastically_driven_dsho_model(δ=0.2, ω=1.0, Q=2.0, σ=0.5; T=Float64)
    β = sqrt(Q^2 - 1 / 4)
    state_transition_matrix =
        T(exp(-ω * δ / 2Q)) * T[
            cos(ω * β * δ / Q) + sin(ω * β * δ / Q) / 2β Q * sin(ω * β * δ / Q) / (ω * β)
            -Q * ω * sin(ω * δ * β / Q) / β cos(ω * δ * β / Q) - sin(ω * δ * β / Q) / 2β
        ]
    observation_matrix = ScalMat{T}(2, 1.0)
    initial_state_mean = zeros(T, 2)
    initial_state_covariance = ScalMat{T}(2, 1.0)
    state_noise_covariance = PDMat{T}(
        Q *
        exp(-ω * δ / Q) *
        [
            ((cos(2ω * δ * β / Q) - 1) - 2β * sin(2ω * δ * β / Q) + 4β^2 * (exp(ω * δ / Q) - 1)) / (8ω^3 * β^2) Q * sin(ω * δ * β / Q)^2 / (2ω^2 * β^2)
            Q * sin(ω * δ * β / Q)^2 / (2ω^2 * β^2) ((cos(2ω * δ * β / Q) - 1) + 2β * sin(2ω * δ * β / Q) + 4β^2 * (exp(ω * δ / Q) - 1)) / (8ω * β^2)
        ],
    )
    observation_noise_covariance = ScalMat{T}(2, σ^2)
    return LinearGaussianModel(
        state_transition_matrix,
        observation_matrix,
        initial_state_mean,
        initial_state_covariance,
        state_noise_covariance,
        observation_noise_covariance,
    )
end

function Assimilate.state(model::LinearGaussianModel)
    return model.state
end

function Assimilate.restore_state!(model::LinearGaussianModel, state)
    model.state ≢ state && (model.state .= state)
    return nothing
end

function Assimilate.step!(model::LinearGaussianModel, t)
    mul!(model.state_buffer, model.state_transition_matrix, model.state, 1.0, 0.0)
    model.state .= model.state_buffer
    return nothing
end

function Assimilate.observation_dimension(model::LinearGaussianModel, t)
    return size(model.observation_matrix, 1)
end

function Assimilate.control_dimension(model::LinearGaussianModel, t)
    if t == 1
        return size(model.initial_state_covariance, 1)
    else
        return size(model.state_noise_covariance, 1)
    end
end

function enzyme_compat_unwhiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)
    # Currently applying forward-mode differentiation through PDMats.unwhiten! using Enzyme
    # hits against an error 'cannot handle blas argument within trmv...' with `trmv`
    # corresponding to the BLAS triangular matrix - vector multiply function - see
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2618 for related issue
    # To work around this we extract triangular Cholesky factor as a dense array and
    # multiply by this instead which incurs an overhead for copying array but still
    # reuses the cached Cholesky factorization to avoid repeatedly refactoring
    return mul!(r, Array(cholesky(a).L), x)
end

function Assimilate.update!(model::LinearGaussianModel, v, t)
    if t == 1
        enzyme_compat_unwhiten!(model.state, model.initial_state_covariance, v)
        model.state .+= model.initial_state_mean
    else
        enzyme_compat_unwhiten!(model.state_buffer, model.state_noise_covariance, v)
        model.state .+= model.state_buffer
    end
    return nothing
end

function Assimilate.observe!(y, model::LinearGaussianModel, t)
    mul!(y, model.observation_matrix, model.state)
    return nothing
end

function Assimilate.observation_covariance(model::LinearGaussianModel{T}, t) where {T}
    return model.observation_noise_covariance
end

function Assimilate.control_covariance(model::LinearGaussianModel{T}, t) where {T}
    return ScalMat{T}(Assimilate.control_dimension(model, t), 1.0)
end
