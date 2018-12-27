#!/usr/

## Collects all the files for the TB (TIght-binding) module

module TB

#using Plots


using Docile
@docstrings


include("TBsparse.jl")
include("TBAux.jl")
include("TBSite.jl")
include("TBqm_frc.jl")
include("TBgeom.jl")
include("TBoptim.jl")
include("TBmultiscale.jl")
include("TBbenchmark.jl")

include("TB_QMMMen_Solver.jl")

end

