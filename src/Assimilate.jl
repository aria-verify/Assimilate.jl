"Julia implementations of data assimilation algorithms."
module Assimilate

using DocStringExtensions

include("model.jl")

"Abstract base type for data assimilation methods."
abstract type AbstractAssimilationMethod end

end
