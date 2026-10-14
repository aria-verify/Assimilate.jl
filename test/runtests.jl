using Assimilate
using Test
using Aqua

@testset "Assimilate.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Assimilate)
    end
    # Write your tests here.
end
