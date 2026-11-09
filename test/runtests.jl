using Assimilate
using Enzyme
using Test
using Aqua
using Random

const RANDOM_SEED = 21983612651875

include("nsks_model.jl")
include("lg_model.jl")

function make_test_nsks_ssm(; T=Float64)
    n_timestep_per_obs = 100
    timestep = T(0.001)
    observation_points = collect((0:1:31) * T(π))
    observation_noise_std = T(0.1)
    initial_state_parameters = IsotropicMatern(;
        length_scale=one(T), output_scale=T(1.0), smoothness=T(4.5)
    )
    model_error_parameters = IsotropicMatern(;
        length_scale=one(T), output_scale=T(0.1), smoothness=T(4.5)
    )
    parameters = NSKSModelParameters{T}()
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

function make_test_ssms(floating_point_types=(Float32, Float64))
    return [
        [make_test_nsks_ssm(; T) for T in floating_point_types];
        [stochastically_driven_dsho_model(; T) for T in floating_point_types]
    ]
end

function setup_model_arrays(
    n_observation_time,
    model::Assimilate.AbstractStateSpaceModel{T},
    rng;
    include_w_and_y=true,
) where {T}
    v_seq = [
        randn(rng, T, Assimilate.control_dimension(model, t)) for t in 1:n_observation_time
    ]

    x_seq = Vector{typeof(Assimilate.state(model))}(undef, n_observation_time)
    ȳ_seq = [
        fill(T(NaN), Assimilate.observation_dimension(model, t)) for
        t in 1:n_observation_time
    ]
    if !include_w_and_y
        return v_seq, x_seq, ȳ_seq
    else
        w_seq = [
            randn(rng, T, Assimilate.observation_dimension(model, t)) for
            t in 1:n_observation_time
        ]
        y_seq = [
            fill(T(NaN), Assimilate.observation_dimension(model, t)) for
            t in 1:n_observation_time
        ]
        return v_seq, w_seq, x_seq, ȳ_seq, y_seq
    end
end

@testset "Assimilate.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Assimilate)
    end
    @testset "$(
        "Model interface with $(summary(model)) for T=$(n_observation_time)"
    )" for n_observation_time in [1, 10], model in make_test_ssms()
        T = eltype(model)
        @test T <: Number
        for t in 1:n_observation_time
            dim_v = Assimilate.control_dimension(model, t)
            @test dim_v isa Int && dim_v > 0
            dim_y = Assimilate.observation_dimension(model, t)
            @test dim_y isa Int && dim_y > 0
            cov_v = Assimilate.control_covariance(model, t)
            @test cov_v isa AbstractMatrix &&
                size(cov_v) == (dim_v, dim_v) &&
                isposdef(cov_v) &&
                eltype(cov_v) === T
            cov_y = Assimilate.observation_covariance(model, t)
            @test cov_y isa AbstractMatrix &&
                size(cov_y) == (dim_y, dim_y) &&
                isposdef(cov_y) &&
                eltype(cov_y) === T
        end
        initial_state = deepcopy(Assimilate.state(model))
        rng = Xoshiro(RANDOM_SEED)
        v_seq, w_seq, x_seq, ȳ_seq, y_seq = setup_model_arrays(
            n_observation_time, model, rng
        )
        v_seq_original, w_seq_original = deepcopy(v_seq), deepcopy(w_seq)
        Assimilate.simulate!(model, v_seq, w_seq, x_seq, ȳ_seq, y_seq)
        final_state = Assimilate.state(model)
        @test initial_state != final_state
        @test x_seq[1] == initial_state
        @test all(v_seq .== v_seq_original)
        @test all(w_seq .== w_seq_original)
        ȳ_seq_2 = similar.(ȳ_seq)
        Assimilate.restore_state!(model, initial_state)
        @test Assimilate.state(model) == initial_state
        Assimilate.simulate!(model, v_seq, ȳ_seq_2)
        @test all(ȳ_seq_2 .== ȳ_seq_2)
        nld_1 = Assimilate.negative_log_density(model, v_seq, y_seq, ȳ_seq)
        @test nld_1 isa T
        nld_2 = Assimilate.negative_log_density(model, v_seq, y_seq)
        @test nld_2 isa T
        @test nld_1 ≈ nld_2
    end
    @testset "Test simulate TLM" for n_observation_time in [1, 10]
        model_1 = stochastically_driven_dsho_model()
        model_2 = deepcopy(model_1)
        rng = Xoshiro(RANDOM_SEED)
        v_seq_1, x_seq_1, ȳ_seq_1 = setup_model_arrays(
            n_observation_time, model_1, rng, include_w_and_y=false
        )
        dmodel_1 = Enzyme.make_zero(model_1)
        dv_seq_1 = [
            randn(rng, eltype(model_1), Assimilate.control_dimension(model_1, t)) for
            t in 1:n_observation_time
        ]
        dx_seq_1 = Enzyme.make_zero(x_seq_1)
        dȳ_seq_1 = Enzyme.make_zero(ȳ_seq_1)
        Enzyme.autodiff(
            Forward,
            Assimilate.simulate!,
            Const,
            Duplicated(model_1, dmodel_1),
            Duplicated(v_seq_1, dv_seq_1),
            Duplicated(x_seq_1, dx_seq_1),
            Duplicated(ȳ_seq_1, dȳ_seq_1),
        )
        v_seq_2 = deepcopy(v_seq_1)
        dv_seq_2 = deepcopy(dv_seq_1)
        x_seq_2 = similar.(x_seq_1)
        ȳ_seq_2 = similar.(ȳ_seq_1)
        dmodel_2 = Enzyme.make_zero(model_2)
        dȳ_seq_2 = Enzyme.make_zero(ȳ_seq_2)
        Assimilate.tlm_simulate!(
            model_2, dmodel_2, v_seq_2, dv_seq_2, x_seq_2, ȳ_seq_2, dȳ_seq_2
        )
        @test maximum(norm.(dȳ_seq_1 .- dȳ_seq_2, Inf)) < 1e-12
    end
    @testset "Test log density gradient" for n_observation_time in [1, 10]
        model = stochastically_driven_dsho_model()
        rng = Xoshiro(RANDOM_SEED)
        v_seq, w_seq, x_seq, ȳ_seq, y_seq = setup_model_arrays(
            n_observation_time, model, rng
        )
        Assimilate.simulate!(model, v_seq, w_seq, x_seq, ȳ_seq, y_seq)
        dmodel = Enzyme.make_zero(model)
        dv_seq = Enzyme.make_zero(v_seq)
        dy_seq = Enzyme.make_zero(y_seq)
        Enzyme.autodiff(
            Reverse,
            Assimilate.negative_log_density,
            Active,
            Duplicated(model, dmodel),
            Duplicated(v_seq, dv_seq),
            Duplicated(y_seq, dy_seq),
        )
        Assimilate.restore_state!(model, x_seq[1])
        Enzyme.remake_zero!(dmodel)
        dv_seq_2 = Enzyme.make_zero(v_seq)
        dȳ_seq = Enzyme.make_zero(ȳ_seq)
        Assimilate.grad_negative_log_density!(
            model, dmodel, v_seq, dv_seq_2, x_seq, ȳ_seq, dȳ_seq, y_seq
        )
        @test maximum(norm.(dv_seq .- dv_seq_2, Inf)) < 1e-12
    end
end
