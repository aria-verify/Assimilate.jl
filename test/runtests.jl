using Varia
using Test
using Aqua
using JET

@testset "Varia.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Varia)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Varia; target_defined_modules = true)
    end
    # Write your tests here.
end
