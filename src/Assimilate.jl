"Julia implementations of data assimilation algorithms."
module Assimilate

using LinearAlgebra
using DocStringExtensions

include("model.jl")

"Abstract base type for data assimilation methods."
abstract type AbstractAssimilationMethod end

include("variational.jl")

end
