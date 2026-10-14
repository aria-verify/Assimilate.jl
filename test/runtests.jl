using Assimilate
using Test
using Aqua
using Random

const RANDOM_SEED = 21983612651875

include("nsks_model.jl")

function make_test_ssm()
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

@testset "Assimilate.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Assimilate)
    end
    @testset "Model interface for T=$(n_observation_time)" for n_observation_time in [1, 10]
        model = make_test_ssm()
        @test Assimilate.control_dimension(model, 1) isa Int
        @test Assimilate.control_dimension(model, 1) > 0
        @test Assimilate.observation_dimension(model, 1) isa Int
        @test Assimilate.observation_dimension(model, 1) > 0
        rng = Xoshiro(RANDOM_SEED)
        v_seq = [
            randn(rng, Assimilate.control_dimension(model, t)) for t in 1:n_observation_time
        ]
        x_seq = Vector{typeof(Assimilate.state(model))}(undef, n_observation_time)
        ȳ_seq = [
            zeros(Assimilate.observation_dimension(model, t)) for t in 1:n_observation_time
        ]
        Assimilate.simulate!(x_seq, ȳ_seq, model, v_seq)
        @test any(any(y .!= 0) for y in ȳ_seq)
        y_seq = [
            ȳ + randn(rng, Assimilate.observation_dimension(model, t)) for
            (t, ȳ) in enumerate(ȳ_seq)
        ]
        nld = Assimilate.negative_log_density(y_seq, ȳ_seq, v_seq, model)
        @test nld > 0
    end
end
