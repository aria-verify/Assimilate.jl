using Assimilate
using Test
using Aqua
using Random

const RANDOM_SEED = 21983612651875

include("nsks_model.jl")
include("lg_model.jl")

function make_test_nsks_ssm()
    n_timestep_per_obs = 100
    timestep = 0.001
    observation_points = collect((0:1:31) * π)
    observation_noise_std = 0.1
    initial_state_parameters = IsotropicMatern(;
        length_scale=1.0, output_scale=1.0, smoothness=4.5
    )
    model_error_parameters = IsotropicMatern(;
        length_scale=1.0, output_scale=0.1, smoothness=4.5
    )
    parameters = NSKSModelParameters()
    model = NSKSModel(parameters)
    return NSKSStateSpaceModel(
        model,
        n_timestep_per_obs,
        timestep,
        observation_points,
        observation_noise_std,
        initial_state_parameters,
        model_error_parameters,
    )
end

function setup_model_arrays(n_observation_time, model, rng)
    v_seq = [
        randn(rng, Assimilate.control_dimension(model, t)) for t in 1:n_observation_time
    ]
    w_seq = [
        randn(rng, Assimilate.observation_dimension(model, t)) for t in 1:n_observation_time
    ]
    x_seq = Vector{typeof(Assimilate.state(model))}(undef, n_observation_time)
    ȳ_seq = [
        zeros(Assimilate.observation_dimension(model, t)) for t in 1:n_observation_time
    ]
    y_seq = [
        zeros(Assimilate.observation_dimension(model, t)) for t in 1:n_observation_time
    ]
    return v_seq, w_seq, x_seq, ȳ_seq, y_seq
end

@testset "Assimilate.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Assimilate)
    end
    @testset "$(
        "Model interface with $(summary(model)) for T=$(n_observation_time)"
    )" for n_observation_time in [1, 10],
        model in [make_test_nsks_ssm(), stochastically_driven_dsho_model()]

        @test Assimilate.control_dimension(model, 1) isa Int
        @test Assimilate.control_dimension(model, 1) > 0
        @test Assimilate.observation_dimension(model, 1) isa Int
        @test Assimilate.observation_dimension(model, 1) > 0
        rng = Xoshiro(RANDOM_SEED)
        v_seq, w_seq, x_seq, ȳ_seq, y_seq = setup_model_arrays(
            n_observation_time, model, rng
        )
        Assimilate.simulate!(model, v_seq, w_seq, x_seq, ȳ_seq, y_seq)
        @test any(any(y .!= 0) for y in ȳ_seq)
        nld = Assimilate.negative_log_density(model, v_seq, y_seq, ȳ_seq)
        @test nld > 0
    end
end
