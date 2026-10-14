using Assimilate
using Test
using Aqua
using JET

@testset "Assimilate.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Assimilate)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Assimilate; target_defined_modules = true)
    end
    # Write your tests here.
end
