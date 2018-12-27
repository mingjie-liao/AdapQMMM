

## Include auxiliary files and libraries
# meshgrid
include("Misc.jl")




## tbgeom
# =========
# type that collects the geometry information for a TB simulation.
#
# CONVENTION HOW THIS IS USED:
#   * each particle considered in the simulation has an entry in the
#     d-dimensional "index-array" `i`. the index I[i,j,k] identifies where in
#     the X and P arrays the particle information is stored. E.g.,
#        X[:, I[o]] is the position of the "origin particle".
#        X[:, I[o+r]] is the position of a particle at o+r in the reference grid
#   * While it is not necessarily fixed what X stands for, it will normally
#     be some reference configuration, whereas displacements or deformed
#     configurations are just stored as arrays, independently of this
#     tbgeom structure.
#   * P[:, i] is a property array. This is used, e.g., to determine whether
#     a particle is in the QM region, in the MM region, in the boundary region,
#     etc.
#   * Finally, if I[i,j,l] <= 0, then this means that this grid point does not
#     contain a corresponding particle.
#   * The array X2I contains the reverse information: X2I[:, j] given the index
#     d-tuple of X[:,j] in I; i.e., X[:, I[X2I[:, j]] == X[:, j].
#   * To iterate over all particles, it is most convinient to iterate over
#     the X array.
#   * the `info` Dict can store any amount of additional information that
#     is not performance critical, but may be useful to have
#

type tbgeom{DIM}
    
    I::Array{Int, DIM}   # a L1 x L2 x L3 Z^d grid; each entry is an Int32
                         # which "points" to a position in grid. This is in
                         # essence connectivity information
    X::Array{Float64,2}  # dim x nX  positions of particles
    X2I::Array{Int,2}    # dim x nX  index vectors of particles in I.

    nX::Int            # number of atoms in the X array (not in the I array)
    dim::Int           # space dimension (domain, i.e. size of I)
                       # future versions could have X = rgDim x nX

    info::Dict
    
end


# default constructor. On purpose, we leave all fields initialised trivially
# so that an immediate error occurs when one tries to access one.
# the only initialised fields are info - for convenience, and I,
# because this determines the type
function tbgeom{DIM}(I::Array{Int, DIM})
    z2 = zeros(0,0)
    return tbgeom{DIM}(I, z2, z2, -1, -1, Dict())
end


## some methods to access geom.I more conveniently
## WARNING: THE FOLLOWING METHODS ARE DEPRECATED AND TO BE REPLACED
##          WITH  THE ONES BELOW
#function setI!{T}(a::AbstractArray{T, 2}, val, idx)
#    a[idx[1], idx[2]] = val
#end
#function setI!{T}(a::AbstractArray{T, 3}, val, idx)
#    a[idx[1], idx[2], idx[3]] = val
#end
#getI{T}(a::AbstractArray{T, 2}, idx) =
#    a[idx[1], idx[2]]
#getI{T}(a::AbstractArray{T, 2}, L, iL) =
#    a[L[1,iL], L[2,iL]]
#getI{T}(a::AbstractArray{T, 2}, L, iL, R, iR) =
#    a[ L[1,iL]+R[1,iR], L[2,iL]+R[2,iR] ]



function setI!(geom::tbgeom{2}, val, idx)
    geom.I[idx[1], idx[2]] = val
end
function setI!(geom::tbgeom{3}, val, idx)
    geom.I[idx[1], idx[2], idx[3]] = val
end
getI(geom::tbgeom{2}, idx) = geom.I[idx[1], idx[2]]
getI(geom::tbgeom{2}, L, iL, R, iR) =
    geom.I[ L[1,iL]+R[1,iR], L[2,iL]+R[2,iR] ]
getI(geom::tbgeom{3}, idx) = geom.I[idx[1], idx[2], idx[3]]
getI(geom::tbgeom{3}, L, iL, R, iR) =
    geom.I[ L[1,iL]+R[1,iR], L[2,iL]+R[2,iR], L[3,iL]+R[3,iR] ]




# # index magic
# getindex{T}(a::AbstractArray{T,2}, idx::Array{Integer,1}) = a[idx[1], idx[2]]
# getindex{T}(a::AbstractArray{T,3}, idx::Array{Integer,1}) = a[idx[1], idx[2], idx[3]]
# function setindex!{T}(a::AbstractArray{T,2}, x::T, idx::Array{Integer,1})
#     a[idx[1], idx[2]] = x
# end
# function setindex!{T}(a::AbstractArray{T,3}, x::T, idx::Array{Integer,1})
#     a[idx[1], idx[2], idx[3]] = x
# end





# overloading access to tbgeom.info, by allowing
#    geom["blah"] instead of geom.info["blah"]
# ML: may need to import Base
import Base.getindex 
import Base.setindex!
function getindex(geom::tbgeom, ii)
    return geom.info[ii]
end
function setindex!(geom::tbgeom, x, ii)
    geom.info[ii] = x
end



# helper functions to recompute the I array from the X2I array
function recompute_I!(geom::tbgeom)
    # first, clean I
    fill!(geom.I, 0.0)
    # now loop through X2I
    for n = 1:size(geom.X2I, 2)
        setI!(geom, n, geom.X2I[:, n])
    end
end



# helper function, creating a zero-cube
function zero_cube(L, dim)
    if dim == 1
        return zeros(Int, L)
    elseif dim == 2
        return zeros(Int, L, L)
    elseif dim == 3
        return zeros(Int, L, L, L)
    else
        error("TBgeom : zero_cube : requires dim \in \{1,2,3\}")
    end
end


# function tbgeom(A, R)
#   a tbgeom  constructor that generates a tbgeom with underlying deformation
#   matrix A and approximate domain radius R
function tbgeom(A, R)
    # start the tbgeom
    dim = size(A, 1)
    # get smallest singular value
    sig = minimum(svd(A)[2])
    # generate a cubic portion of Zd containing -R/sig : R/sig in each
    # coordinate direction
    ndim = ceil(Int, R/sig)
    I = zero_cube(2*ndim+1, dim)
    # initialise and start filling `geom`
    geom = tbgeom(I)
    geom.dim = dim
    geom["A"] = A
    
    # obtain the deformed point list
    Z = dgrid_list(-ndim:ndim, geom.dim)
    X = A * Z
    geom["X0"] = X
    # find points inside the ball and return them
    I_ball = find( sumabs2(X, 1) .< R^2 )
    geom.nX = length(I_ball)
    geom.X = X[:, I_ball]
    # geom.X2I = int16(Z[:, I_ball]) + ndim + 1
	# geom.X2I = Int64(Z[:, I_ball]) + ndim + 1
	geom.X2I = convert(Array{Int}, Z[:, I_ball]) + ndim + 1

    # construct the I list
    recompute_I!(geom)

    # define the origin
    geom["oI"] = (ndim+1)*ones(Int, geom.dim)
    oI = geom["oI"]
    geom["oX"] = getI(geom, geom["oI"])
    geom["o"] = geom.X[:, getI(geom, oI)]
    
    # return the geom object
    return geom
end  #  function tbgeom(A, Rqm; Rbuf = 0, Rmm = 0)


################# Screw dislocation in JuLIPMaterials #######################

function geom_screw!(geom::tbgeom; b = 1.0)
    
    # shift the lattice so the origin is in the centre of a triangle   
    x0 = [0.0, 0.0]
    _ , I0 = findmin( sumabs2(geom.X .- x0, 1) )
    tcore = [0.5, √3/6]
    xc = geom.X[:, I0] + tcore
    X = geom.X .- xc
    geom.X = copy(X)
    
    # # apply dislocation FF predictor
    geom_screw_predictor!(geom; b = b)
    # if edgevacancy
    #      X = geom.X .+ tcore
    # end
    # geom.X = copy(X)
    return geom

end

function geom_screw_predictor!(geom::tbgeom; b = 1.0)
    x0 = 1/2 
    y0 = √3/6
    z = (geom.X[1, :] - x0) + im * (geom.X[2, :] - y0)
    θ = angle(z)
    geom.X = [geom.X; 1/(2*π) * b * θ]
    return geom
end
################################################################################


################# Edge dislocation in JuLIPMaterials #######################

function geom_edge!(geom::tbgeom; b = 1.0, ν = 0.25, xicorr = true, edgevacancy = true)
    
    #x0 = [0.5, √3/2]
    x0 = [0.0, 0.0]
    _ , I0 = findmin( sumabs2(geom.X .- x0, 1) )
    tcore = [0.5, √3/6]
    xc = geom.X[:, I0] + tcore
    # println(I0, xc)
    # shift configuration to move core to 0
    X = geom.X .- xc
    geom.X = copy(X)
    
    # remove the center-atom
    if edgevacancy
       geom_vacancies!(geom, geom["oI"])  
    end
    # # apply dislocation FF predictor
    geom_edge_predictor!(geom; b = b, xicorr = xicorr, ν = ν)
    # # @show geom.X
    # if edgevacancy
         X = geom.X .+ tcore
    # end
    # # println(X)
    geom.X = copy(X)
    return geom

end

"""
standard isotropic CLE edge dislocation solution
"""
function ulin_edge_isotropic(X, b, ν)
    x, y = X[1, :], X[2, :]
    r² = x.^2 + y.^2
    ux = b/(2*π) * ( angle.(x + im*y) + (x .* y) ./ (2*(1-ν) * r²) )
    uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log.(r²) + - 2 * y.^2 ./ (4*(1-ν) * r²) )
    return [ux'; uy']
end

"""
lattice corrector to CLE edge solution; cf EOS paper
"""
function xi_solver(Y::Vector, b; TOL = 1e-10, maxnit = 5)
    
    ξ1(x::Real, y::Real, b) = x - b * angle.(x + im * y) / (2*π)
    dξ1(x::Real, y::Real, b) = 1 + b * y / (x^2 + y^2) / (2*π)
    
    y = Y[2]
    x = y
    for n = 1:maxnit
        f = ξ1(x, y, b) - Y[1]
        if abs(f) <= TOL; break; end
        x = x - f / dξ1(x, y, b)
    end
    if abs(ξ1(x, y, b) - Y[1]) > TOL
        warn("newton solver did not converge at Y = $Y; returning input")
        return Y
    end
    return [x, y]
end

"""
EOSShapeev edge dislocation solution
"""
function ulin_edge_eos(X, b, ν)
    Xmod = zeros(2, size(X, 2))
    for n = 1:size(X, 2)
        Xmod[:, n] = xi_solver(X[1:2, n], b)
    end
    return ulin_edge_isotropic(Xmod, b, ν)
end

function geom_edge_predictor!(geom::tbgeom; b = 1.0, xicorr = true, ν = 0.25)
   X = copy(geom.X)
   if xicorr
      X[1:2, :] += ulin_edge_eos(X, b, ν)  # always use the corrector
   else
      X[1:2, :] += ulin_edge_isotropic(X, b, ν)
   end
   geom.X = X
   return geom
end

################################################################################



# function geom_vacancies!(geom, vList)
#
#   remove some atoms from the lattice. The list of vacancy sites, `vList` may
#   be given either as a list of X-indixes (scalar) or I-indices (vectors)
function geom_vacancies!(geom::tbgeom, vList)

    # extract X-indices if vList is passed as I-indices
    if size(vList, 1) == geom.dim
        vList_ = vList
        vList = zeros(Int, size(vList_, 2))
        for n = 1:size(vList_, 2)
            vList[n] = getI(geom, vList_[:, n])
        end
    end
        
    # remove the vList indices from X and from X2I
    keepList = setdiff(1:geom.nX, vList)
    geom.X = geom.X[:, keepList]
    geom.X2I = geom.X2I[:, keepList]
    geom.nX -= length(vList)

    # recompute the I array
    recompute_I!(geom)

end

# return the origin
function origin(geom::tbgeom)
    try o = geom.info["o"]
    catch o = zeros(geom.dim)
    end
end


# function geom_defineregions!(geom::tbgeom;
#                              Rqm=0, buffer_width = 0, bc_width = 0)
#
#    function to setup the QM/MM and buffer regions
#    and corresponding index sets
#
#    If Rqm == Inf, then the entire region is QM; the bc_width is then
#    taken as max(bc_width, buffer_width)
#
function geom_defineregions!(geom::tbgeom;
                             Rqm=0, buffer_width = 0,
                             bc_width = 0, stencil_width = 0, Rref = 0)

    # compute distances from origin
    # tcore = [0.5, √3/6]
    o = origin(geom)# .- 1/2*tcore 
    r2 = sumabs2(geom.X - o * ones(1, geom.nX), 1)
    Rmax = sqrt(maximum(r2))

    # If Rqm == Inf, then this means the entire simulation is quantum
    if Rqm == Inf
        Rfree = Rmax - max(buffer_width, bc_width)
        geom["iQM+Buf"] = 1:geom.nX
        geom["iQM"] = find(r2 .≤ Rfree^2)
        geom["iBC"] = setdiff(1:geom.nX, geom["iQM"])
        geom["iMM"] = []
        # add for APE 
        geom["iAPE"] = setdiff(find(r2 .<= Rref^2), geom["iQM"])
    else
        # Rmm is obtained from the domain size and the bc_width
        Rbuf = Rqm + buffer_width
        Rmm = Rmax - bc_width
    
        # identify the different regions
        geom["iQM"] = find(r2 .<= Rqm^2)
        geom["iQM+Buf"] = find(r2 .<= Rbuf^2)
        geom["iMM"] =  setdiff(find(r2 .<= Rmm^2), geom["iQM"])
        geom["iBC"] = find(r2 .> Rmm^2)
        # TODO: this is getting a bit too much - need to find a better
        #       way to prepare the geometry without storing quite so
        #       much information
        # iMME is the list of sites over which to sum the site-energy!
        geom["iMME"] = setdiff(find(r2 .<= (Rmm+stencil_width)^2), geom["iQM"])
    end
end


# function plot(geom; Y = [])
#  plotting/visualising a tbgeom object
# function plot(geom::tbgeom; Y=[], style="k.")
# using PyPlot
import PyPlot
function plot(geom::tbgeom; Y=[], style="k.")
#	import PyPlot
#	using PyPlot
    # # default, if no configuration has been passed?
    if isempty(Y)
        Y = geom.X
    end

    try
        iQM = geom["iQM"]
        iBuf = setdiff(geom["iQM+Buf"], iQM)
        iMM = setdiff(geom["iMM"], iBuf)
        iBC = geom["iBC"]
        #iAPE = geom["iAPE"]
        PyPlot.plot(Y[1,iQM], Y[2,iQM], "r.", markersize=10)
        PyPlot.plot(Y[1,iBuf], Y[2,iBuf], "m.", markersize=10)
        PyPlot.plot(Y[1,iMM], Y[2,iMM], "b.", markersize=10)
        PyPlot.plot(Y[1,iBC], Y[2,iBC], "g.", markersize=10)
        #PyPlot.plot(Y[1,iAPE], Y[2,iAPE], "m.")
    catch        
        PyPlot.plot(Y[1,:]', Y[2,:]', style)
    end
	nothing
end

# import Plots
# plotlyjs()
function plotz(geom::tbgeom; Y=[], style="k.")
#	import PyPlot
	# using Plots
    # # default, if no configuration has been passed?
    if isempty(Y)
        Y = geom.X
    end

    try
        iQM = geom["iQM"]
        iBuf = setdiff(geom["iQM+Buf"], iQM)
        iMM = setdiff(geom["iMM"], iBuf)
        iBC = geom["iBC"]
        #iAPE = geom["iAPE"]
        Plots.scatter(Y[1,iQM], Y[2,iQM], Y[3,iQM], color=:red, marker=([:hex],1.5))
        Plots.scatter!(Y[1,iBuf], Y[2,iBuf], Y[3,iBuf], color=:pink, marker=([:hex],1.5))
        Plots.scatter!(Y[1,iMM], Y[2,iMM], Y[3,iMM], color=:blue, marker=([:hex],1.5))
        Plots.scatter!(Y[1,iBC], Y[2,iBC], Y[3,iBC], color=:green, marker=([:hex],1.5))
        #PyPlot.plot(Y[1,iAPE], Y[2,iAPE], "m.")
    catch        
        PyPlot.plot(Y[1,:]', Y[2,:]', style)
    end
	nothing
end


# neighbourhood computations
# =============================

# function getNearestNeigRg(geom::tbgeom)
#    This function computes a naive nearest-neighbourhood direction
#    list, containing all \pm e_i vectors.
#    It can be overwritten to replace it by a better list
#
function getNearestNeigRg(geom::tbgeom; Rnn = 1.1)
    if geom.dim == 2
        # TODO: rewrite as NN finder ?
        return [ eye(Int, geom.dim) -eye(Int, geom.dim) ]
    elseif geom.dim == 3
        geom_nn = TB.tbgeom(geom["A"], 1.1)
        oI = geom_nn["oI"]
        TB.geom_vacancies!(geom_nn, oI)
        return geom_nn.X2I .- oI
    end
    error("TB.getNearestNeigRg works only for dim 1, 2")
end

# function getNeighbourhood(geom::tbgeom, i0, Rg)
#   compute the interaction neighbourhood of one point
function getNeighbourhood(geom::tbgeom, i0, Rg)
    # convert i0 to a point in geom.I
    i0_I = geom.X2I[:, i0]
    # allocate an index list as long as the number of
    # columns of Rg --- each columns of Rg corresponds to
    # a finite-difference vector.
    Ineigs = zeros(Int, size(Rg, 2))
    # lookup the neighbours from the I-array
    for n = 1:length(Ineigs)
        Ineigs[n] = getI(geom, round(Int, i0_I+Rg[:,n]))
    end
    # return computed list
    return Ineigs
end




#=====================================================
#      Some Helper Functions that eventually
#      will need to be sorted into other 
#      files maybe
# =====================================================#

# function extend(geom_lg:tbgeom, geom:tbgeom, U)
#     extend (geom, U) by zeros to become a function on geom_lg
#     requires that geom_lg is a "superset" of geom

# TODO: check that the two geometries have the same underlying lattice
#
function extend(geom_lg::tbgeom, geom::tbgeom, U)
    # allocate a zero-function
    U_lg = zeros(geom_lg.dim, geom_lg.nX)
    #U_lg = copy(geom_lg.X)
    # get the I origins of the two geometries
    oI = geom["oI"]                 
    oI_lg = geom_lg["oI"]
    # loop through all datapoints
    for n = 1:geom.nX
        # get the grid-index of X[:,n] in geom, calculated
        # relative to the geom-origin
        I_relative = geom.X2I[:, n] - oI
        # compute grid-index of this point in geom_lg, obtained
        # by just shifting w.r.t. the geom_lg-origin
        I_lg = I_relative + oI_lg
        # get the X-index in geom_lg
        n_lg = getI(geom_lg, I_lg)
        # write U[n] into the extended U_lg
        U_lg[:, n_lg] = U[:, n]
    end
    return U_lg
end


# function get_r(geom)
#
#   return a vector containing the distance of points to origin
    
function get_r(geom)
    return sqrt(sumabs2(geom.X .- geom["o"], 1))
end

##################################################
# Here we also need the QM region's index 
# so the following maybe wrong
#function Myextend(geom_lg::tbgeom, geom::tbgeom, U::Array{Float64,2})
#    # allocate a zero-function
#    # U_lg = zeros(geom_lg.dim, geom_lg.nX)
#    U_lg = copy(geom_lg.X)
#    # initiate a zero-function for index
#    idx = Array{Int64, 1}(geom.nX)
#    # get the I origins of the two geometries
#    oI = geom["oI"]                 
#    oI_lg = geom_lg["oI"]
#    # loop through all datapoints
#    for n = 1:geom.nX
        # get the grid-index of X[:,n] in geom, calculated
        # relative to the geom-origin
#I_relative = geom.X2I[:, n] - oI
#        # compute grid-index of this point in geom_lg, obtained
        # by just shifting w.r.t. the geom_lg-origin
#        I_lg = I_relative + oI_lg
#        # get the X-index in geom_lg
#        n_lg = getI(geom_lg, I_lg)
 #       # write U[n] into the extended U_lg
  #      U_lg[:, n_lg] = U[:, n]
        # store the idx
#        idx[n] = copy(n_lg)
#    end
#    return U_lg, idx
#end

# we add the QM regin index
function Myextend(geom_lg::tbgeom, geom::tbgeom, U::Array{Float64,2})
    # allocate a zero-function
    # U_lg = zeros(geom_lg.dim, geom_lg.nX)
    U_lg = copy(geom_lg.X)
    # initiate a zero-function for index
    idx = Array{Int64, 1}(geom.nX)
    idxqm = Array{Int64, 1}(length(geom["iQM"]))
    l = 1
    # get the I origins of the two geometries
    oI = geom["oI"]                 
    oI_lg = geom_lg["oI"]
    # get iQM 
    iQM = geom["iQM"]
    # loop through all datapoints
    for n = 1:geom.nX
        # get the grid-index of X[:,n] in geom, calculated
        # relative to the geom-origin
        I_relative = geom.X2I[:, n] - oI
        # compute grid-index of this point in geom_lg, obtained
        # by just shifting w.r.t. the geom_lg-origin
        I_lg = I_relative + oI_lg
        # get the X-index in geom_lg
        n_lg = getI(geom_lg, I_lg)
        # write U[n] into the extended U_lg
        U_lg[:, n_lg] = U[:, n]
        # store the idx
        idx[n] = copy(n_lg)
    end
    
    for i in iQM
        # get the grid-index of X[:,n] in geom, calculated
        # relative to the geom-origin
        I_relative_qm = geom.X2I[:, i] - oI
        # compute grid-index of this point in geom_lg, obtained
        # by just shifting w.r.t. the geom_lg-origin
        I_lg_qm = I_relative_qm + oI_lg
        # get the X-index in geom_lg
        n_lg_qm = getI(geom_lg, I_lg_qm)  
        # store the idx
        idxqm[l] = copy(n_lg_qm)
        l += 1
    end
    
    return U_lg, idx, idxqm
end


function extendI(geom_lg::tbgeom, geom::tbgeom, I)
    # allocate a zero-function
    # U_lg = zeros(geom_lg.dim, geom_lg.nX)
    Ilg = copy(I)
    # get the I origins of the two geometries
    oI = geom["oI"]                 
    oI_lg = geom_lg["oI"]
    # loop through all datapoints
    for i = 1:length(I)
        # get the grid-index of X[:,n] in geom, calculated
        # relative to the geom-origin
        I_relative = geom.X2I[:, I[i]]- oI
        # compute grid-index of this point in geom_lg, obtained
        # by just shifting w.r.t. the geom_lg-origin
        I_lg = I_relative + oI_lg
        # get the X-index in geom_lg
        n_lg = getI(geom_lg, I_lg)
        # store the idx
        Ilg[i] = copy(n_lg)
    end
    return Ilg
end