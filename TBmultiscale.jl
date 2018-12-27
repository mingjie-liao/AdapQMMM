
#
# TBmultiscale.jl
#
# This implements the main tools needed to make QM/MM coupling work
# with the TB module.
#


@doc """
linear site polynomial
stencil: subset of Z^d
coeffs: dim x dim x (number of stencil entries)
""" ->
type SiteLinear
    stencil::Array{Int, 2}
    coeffs::Array{Float64, 3}
end



# @doc """
# Creates a SiteLinear (linear SitePolynomial), which approximates the force near
# the reference domain A * Z^d.

# INPUT
#   params::tbparams : the parameters of the TB model
#   Aref : d x d matrix defining the reference domain
#   R : radius for the SitePolynomial range
#   Rbuf : buffer radius for the TB calculation

# OUTPUT
#   SiteLinear object
# """ -> 
function siteForceApproximation(params::tbparams, Aref, R, Rbuf)
    # create a geometry with radius R + Rbuf
    geom = tbgeom(Aref, R+Rbuf)

    # compute hessian in the reference state
    idxOrigin = geom["oX"]
    hE = totalE_fd_hess(geom.X, params; i0 = idxOrigin)

    # find the points inside the radius R
    # we put i0 itself at the very end so that we can find the origin again
    r = get_r(geom)
    iR = [setdiff(find(r .<= R), [idxOrigin]); idxOrigin]
    stencil = geom.X2I[:, iR] .- geom["oI"]

    # TODO: the following construction makes an exponentially small error
    # that is nowhere accounted for. Is there a way to somehow average it into
    # the linear polynomial to get the average right? Maybe there is a way
    # to do this to high accuracy using a different method? Or maybe
    # this is just not very relevant.

    # the reduced stencil
    coeff = hE[:, :, iR]
    # but it needs to be corrected to sum to zero
    # WARNING: this assumes Bravais lattice!
    coeff[:, :, end] = 0
    coeff[:, :, end] = -sum(coeff, 3)

    # return a linear site polynomial, i.e. a SiteLinear
    return SiteLinear(stencil, coeff)
end


@doc """
INPUT
   sp::SiteLinear : site polynomial
   geom : tbgeom : description of the lattice geometry
   Y : (dim x geom.nX) list of deformed atom positions
   J : list of atom indices for which the SiteFunction should be evaluated
       (for now stick with length(J) = 1)
RETURN
   value of the linear functional
""" ->
function evalSiteFunction(sp::SiteLinear, geom::tbgeom, Y,  J)
    # warning(" evalSiteFunction : this is slow and should probably not be used ")

    # extract dimension information
    d1, d2, nneig = size(sp.coeffs)
    if d2 != size(Y, 1)
        error("evalSiteFunction : dimensions do not match")
    end

    # initialise the output
    f = zeros(d1)

    # get the X-index and I-index of the "current" center-site
    # (because we are just evaluating at one site, "current" means
    #    this one site)
    jX = J[1]
    jI = geom.X2I[:, jX]
    # allocate a scalaar
    kX = 0

    # loop over neighbouring sites
    for n in 1:nneig
        # get the X-index of the current neighbour
        #   easy-read version:
        #   kX_old = getI(geom.I, jI + sp.stencil[:, n])
        # efficient version: (tested against easy-read version)
        kX = getI(geom, geom.X2I, jX, sp.stencil, n)

        # TODO: a sanity test; this should eventually be turned off.
        #      we probably want some debugging facility that turns off some
        #      slow codes
        if kX == 0
            @show (jI, sp.stencil[:,n])
            error("evalSiteFunction : a neighbour does not exist in geom")
        end
        # evaluate the term (devectorized for performance)
        # f[:] += slice(sp.coeffs, :, :, n) * Y[:, kX]
        for i = 1:d1, j = 1:d2
            f[i] = f[i] + sp.coeffs[i,j,n] * (Y[j,kX] - geom.X[j,kX])
        end
    end

    return f
end


# fast in-place vectorised function to evaluate a SiteLinear
function evalSiteFunction!(sp::SiteLinear,
                           geom::tbgeom,
                           Y::Array{Float64,2},
                           J::Array{Int, 1},
                           Frc::Array{Float64,2})

    # extract dimension information (d1, d2 \in \{2, 3\})
    d1, d2, nneig = size(sp.coeffs)
    # assert type of the index set over which we are looping
    for jX in J
        # initialise force to 0
        Frc[:,jX] = 0.0
        # loop over neighbouring sites
        # @inbounds
        for n in 1:nneig
            # get the X-index of the current neighbour
            # @show geom.X2I[:,jX]
            # @show sp.stencil[:,n]
            kX = getI(geom, geom.X2I, jX, sp.stencil, n)
            # evaluate the term (devectorized for performance)
            for i = 1:d1, j = 1:d2
                Frc[i, jX] += sp.coeffs[i, j, n] * (Y[j, kX]-geom.X[j,kX])
            end
        end
    end
end



# quadratic scalar site function, optimised specifically to be
# used as a site energy approximation
type SiteQuadraticE
    # constant
    E0::Float64
    # interaction range
    stencil::Array{Int, 2}
    # linear component
    coeffs1::Array{Float64, 2}
    # quadratic component
    coeffs2::Array{Float64, 4}
end


#
# fill the SiteQuadraticE
#
#  TODO: we probably want to rewrite this in such a way that
#        instead of providing an inner and outer radius, we give an
#        error on the coefficients that we truncate?
#
#        > furthermore, we may want to create two buffer-zones
#        and outer one that is neglected and an inner one that is
#        averaged into the interior?
#
#        > these kind of ideas probably warrant a separate project at some point
#
function constructSiteQuadraticE(params::tbparams, Aref, R, Rbuf)
    # create a geometry with radius R + Rbuf
    geom = tbgeom(Aref, R+Rbuf)

    # compute site energy, its gradient and hessian
    idxOrigin = geom["oX"]
    Es, dEs = compute(geom.X, ("Es", "dEs"), params; I = idxOrigin)
    hEs = siteE_fd_hess(geom.X, params, idxOrigin)

    # =========== CONSTANT TERM ================
    # the constant term is already given by Es

    # =========== LINEAR TERM ==================
    # find the points inside the radius R
    # we remove the origin iterself since SiteQuadratic is understood to be
    # the coefficients for the finite differences
    r = get_r(geom)
    iR = setdiff(find(r .<= R), [idxOrigin])
    stencil = geom.X2I[:, iR] .- geom["oI"]
    coeff1 = dEs[:, iR]

    # =========== QUADRATIC TERM ==================
    coeff2 = hEs[:, iR, :, iR]

    # return a qu site polynomial, i.e. a SiteLinear
    return SiteQuadraticE(Es, stencil, coeff1, coeff2)
end



#
# TODO: decide whether to keep evaluating the energy-difference (i.e.
#       ignoring E0, or whether it would be better to compute
#       the total site energy and do the difference-calculation elsewhere
#
function evalSiteFunction(sp::SiteQuadraticE,
                          geom::tbgeom,
                          Y::Array{Float64,2},
                          J::Array{Int, 1})

    # compute displacement to save some operations
    U = Y - geom.X
    # initialize array of energies
    E = zeros(length(J))

    # extract dimension information
    d1, nneig = size(sp.coeffs1)

    # loop over sites
    #   n : center-site (ℓ)
    #   i : neighbour (ℓ+ρ)
    #   j : another neighbour (ℓ+ς)
    for n = 1:length(J)
        nX = J[n]

        for i = 1:nneig
            # get the X-index of the first neighbour
            iX = getI(geom, geom.X2I, nX, sp.stencil, i)

            # ============  LINEAR COMPONENT ===========
            # compute inner product
            for a = 1:d1
                E[n] += sp.coeffs1[a, i] * (U[a, iX] - U[a, nX])
            end

            # ============  QUADRATIC COMPONENT ===========
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)
                # compute the quadratic form
                for a = 1:d1, b = 1:d1
                  E[n] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX])
                end
            end
        end
    end

    return E
end      # evalSiteFunction




#
#  Derivative of a quadratic site energy
#
function evalSiteFunction_D1(sp::SiteQuadraticE,
                             geom::tbgeom,
                             Y::Array{Float64,2},
                             J::Array{Int, 1})

    # compute displacement to save some operations
    U = Y - geom.X
    # initialize array of energies
    dE = zeros(geom.dim, geom.nX)
    # @show size(dE)

    # extract dimension information
    d1, nneig = size(sp.coeffs1)

    # loop over sites
    #   n : center-site (ℓ)
    #   i : neighbour (ℓ+ρ)
    #   j : another neighbour (ℓ+ς)
    for n = 1:length(J)
        nX = J[n]

        for i = 1:nneig
            # get the X-index of the first neighbour
            iX = getI(geom, geom.X2I, nX, sp.stencil, i)
            # @show iX

            # ============  LINEAR COMPONENT ===========
            # compute inner product
            for a = 1:d1
                # E[n] += sp.coeffs1[a, i] * (U[a, iX] - U[a, nX])
                dE[a, iX] += sp.coeffs1[a, i]
                dE[a, nX] -= sp.coeffs1[a, i]
            end

            # ============  QUADRATIC COMPONENT ===========
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)
                # compute the derivative of the quadratic form
                for a = 1:d1, b = 1:d1
                    # E[n] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX])
                    dE[a, iX] += 0.5 * sp.coeffs2[a,i,b,j] * (U[b,jX]-U[b,nX])
                    dE[a, nX] -= 0.5 * sp.coeffs2[a,i,b,j] * (U[b,jX]-U[b,nX])
                    dE[b, jX] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX])
                    dE[b, nX] -= 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX])
                end
            end    #  for j
        end    # for i
    end    # for n

    return dE
end      # evalSiteFunction_D1



####################### Cubic expansion for dislocation ###############
#* now add cubic component
#* for dislocation computation
type SiteCubicE
    # constant
    E0::Float64
    # interaction range
    stencil::Array{Int, 2}
    # linear component
    coeffs1::Array{Float64, 2}
    # quadratic component
    coeffs2::Array{Float64, 4}
    # cubic component
    coeffs3::Array{Float64, 6}
end

function constructSiteCubicE(params::tbparams, Aref, R, Rbuf)
    # create a geometry with radius R + Rbuf
    geom = tbgeom(Aref, R+Rbuf)

    # compute site energy, its gradient and hessian
    idxOrigin = geom["oX"]

    # very slow step here
    Es, dEs = compute(geom.X, ("Es", "dEs"), params; I = idxOrigin)
    hEs = siteE_fd_hess(geom.X, params, idxOrigin)
    fdhEs = siteE_fd_hess_fd(geom.X, params, idxOrigin)
    # very slow step here

    # =========== CONSTANT TERM ================
    # the constant term is already given by Es

    # =========== LINEAR TERM ==================
    # find the points inside the radius R
    # we remove the origin iterself since SiteQuadratic is understood to be
    # the coefficients for the finite differences
    r = get_r(geom)
    iR = setdiff(find(r .<= R), [idxOrigin])
    stencil = geom.X2I[:, iR] .- geom["oI"]
    coeff1 = dEs[:, iR]

    # =========== QUADRATIC TERM ==================
    coeff2 = hEs[:, iR, :, iR]

    # =========== CUBIC TERM ==================
    coeff3 = fdhEs[:, iR, :, iR, :, iR]

    # return a qu site polynomial, i.e. a SiteLinear
    return SiteCubicE(Es, stencil, coeff1, coeff2, coeff3)
end


# Just evaluate the energy E
function evalSiteFunction(sp::SiteCubicE,
                          geom::tbgeom,
                          Y::Array{Float64,2},
                          J::Array{Int, 1})

    # compute displacement to save some operations
    U = Y - geom.X
    # initialize array of energies
    E = zeros(length(J))

    # extract dimension information
    d1, nneig = size(sp.coeffs1)

    # loop over sites
    #   n : center-site (ℓ)
    #   i : neighbour (ℓ+ρ)
    #   j : another neighbour (ℓ+ς)
    #   k : another neighbour (ℓ+α)
    for n = 1:length(J)
        nX = J[n]

        for i = 1:nneig
            # get the X-index of the first neighbour
            iX = getI(geom, geom.X2I, nX, sp.stencil, i)

            # ============  LINEAR COMPONENT ===========
            # compute inner product
            for a = 1:d1
                E[n] += sp.coeffs1[a, i] * (U[a, iX] - U[a, nX])
            end

            # ============  QUADRATIC COMPONENT ===========
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)
                # compute the quadratic form
                for a = 1:d1, b = 1:d1
                    E[n] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX])
                end
            end # for j

            # ==============  CUBIC COMPONENT =============
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)                
                # this needs a third loop over neighbours
                for k = 1:nneig
                    # X-index of the third neighbour
                    kX = getI(geom, geom.X2I, nX, sp.stencil, k)
                    # compute the cubic form
                    for a = 1:d1, b = 1:d1, c = 1:d1
                        E[n] += 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX]) * (U[c,kX]-U[c,nX])
                    end
                end # for k 
            end # for j
        

        end
    end

    return E
end      # evalSiteFunction


#
#  Derivative of a cubic site energy
#
function evalSiteFunction_D1(sp::SiteCubicE,
                             geom::tbgeom,
                             Y::Array{Float64,2},
                             J::Array{Int, 1})

    # compute displacement to save some operations
    U = Y - geom.X
    # initialize array of energies
    dE = zeros(geom.dim, geom.nX)

    # extract dimension information
    d1, nneig = size(sp.coeffs1)

    # loop over sites
    #   n : center-site (ℓ)
    #   i : neighbour (ℓ+ρ)
    #   j : another neighbour (ℓ+ς)
    #   k : another neighbour (ℓ+α)
    for n = 1:length(J)
        nX = J[n]

        for i = 1:nneig
            # get the X-index of the first neighbour
            iX = getI(geom, geom.X2I, nX, sp.stencil, i)

            # ============  LINEAR COMPONENT ===========
            # compute inner product
            for a = 1:d1
                # E[n] += sp.coeffs1[a, i] * (U[a, iX] - U[a, nX])
                dE[a, iX] += sp.coeffs1[a, i]
                dE[a, nX] -= sp.coeffs1[a, i]
            end

            # ============  QUADRATIC COMPONENT ===========
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)
                # compute the derivative of the quadratic form
                for a = 1:d1, b = 1:d1
                    # E[n] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX]) 
                    dE[a, iX] += 0.5 * sp.coeffs2[a,i,b,j] * (U[b,jX]-U[b,nX])
                    dE[a, nX] -= 0.5 * sp.coeffs2[a,i,b,j] * (U[b,jX]-U[b,nX])
                    dE[b, jX] += 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX])
                    dE[b, nX] -= 0.5 * sp.coeffs2[a,i,b,j] * (U[a,iX]-U[a,nX])
                end
            end    #  for j

            # =============  CUBIC COMPONENT  ============
            # this needs a second loop over neighbours
            for j = 1:nneig
                # X-index of the second neighbour
                jX = getI(geom, geom.X2I, nX, sp.stencil, j)
                # this needs a third loop over neighbours
                for k = 1:nneig
                    # X-index of the third neighbour
                    kX = getI(geom, geom.X2I, nX, sp.stencil, k)
                    # compute the derivative of the cubic form
                    for a = 1:d1, b = 1:d1, c = 1:d1
                        # E[n] += 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX]) * (U[c,kX]-U[c,nX])
                        dE[a, iX] += 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[b,jX]-U[b,nX]) * (U[c,kX]-U[c,nX])
                        dE[a, nX] -= 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[b,jX]-U[b,nX]) * (U[c,kX]-U[c,nX])
                        dE[b, jX] += 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[c,kX]-U[c,nX])
                        dE[b, nX] -= 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[c,kX]-U[c,nX])
                        dE[c, kX] += 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX])
                        dE[c, nX] -= 1/6 * sp.coeffs3[a,i,b,j,c,k] * (U[a,iX]-U[a,nX]) * (U[b,jX]-U[b,nX])
                    end
                end     # for k
            end    #  for j
        
        end    # for i
    end    # for n

    return dE
end      # evalSiteFunction_D1

################################################################

# function QM_MM_frc(Y, tasks, sp, params, geom)
#
# compute the hybrid QM/MM force
#
# Y : positions
# tasks: only "F" allowed at present
# sp : SiteFunction representing a force
# params : tbparams
# geom : tbgeom that has been prepared to do a QM/MM simulation
#

function QM_MM_frc(Y, tasks, sp, params, geom)

    if tasks != "F"
        error(""" QM_MM_frc only admits the task "F" for now""")
    end

    # allocate the return vector
    Frc = zeros(geom.dim, geom.nX)

    # compute the QM component
    iQM_plus_Buf = geom["iQM+Buf"]
    iBuf = setdiff(iQM_plus_Buf, geom["iQM"])
    Y_qm = Y[:, iQM_plus_Buf]
    Frc[:, iQM_plus_Buf] = compute(Y_qm, "dE", params)
    Frc[:, iBuf] = 0.0

    # evaluate the forces in the MM region
    evalSiteFunction!(sp, geom, Y, geom["iMM"], Frc)

    return Frc
end



#
# Energy functional for the energy-based QM/MM scheme
#
function QM_MM_en(Y, tasks, sp, params, geom)

    # REMARK
    # Strictly speaking we should worry about whether to compute just E or just
    # dE or both, but dE is so much more expensive than E that we just won't
    # worry about it.
    # this should probably be improved in future versions, or once we find that
    # it becomes a bottleneck.

    # Assemble everything that is requested
    # force the flags into a tuple, if they are not (i.e. if only one of them
    # has been passed.
    if (typeof(tasks) == String) || (typeof(tasks)==Char)
        tasks = (tasks,)
    end
    # start the return thingy
    ret = ()
    # loop through all tasks
    for cur_task in tasks

        # ASSEMBLE THE ENERGY
        # ====================
        if string(cur_task) == "E"
            E = QM_MM_en_E(Y, sp, params, geom)
            ret = tuple(ret..., E)

        # ASSEMBLE THE FORCES
        # ====================
        elseif string(cur_task) == "dE"
            dE = QM_MM_en_dE(Y, sp, params, geom)
            ret = tuple(ret..., dE)
        end
    end

end



# Evaluation of the energy for energy-based QM/MM coupling
#
#  TODO: SiteQuadraticE  needs to be replaced with something more general!
#
function QM_MM_en_E(Y, sp::Union{SiteQuadraticE,SiteCubicE}, params::tbparams, geom::tbgeom)

    # ============= QM COMPONENT ==============
    Iqmbuf = geom["iQM+Buf"]
    Iqm = findin(Iqmbuf, geom["iQM"])
    E_qm = compute( Y[:, geom["iQM+Buf"]], "Es", params; I=Iqm )
    # subtract the ground state energy
    # E_qm -= length(Iqm) * sp.E0
    
    # ============= MM COMPONENT ==============
    # error(" what set of MM sites used! [TODO]")
    E_mm = sum( evalSiteFunction(sp, geom, Y, geom["iMME"]) )
    
    # return the sum of the two
    return E_qm + E_mm
end


#
# Evaluation of forces for energy-based QM/MM coupling
#
function QM_MM_en_dE(Y, sp::Union{SiteQuadraticE,SiteCubicE}, params::tbparams, geom::tbgeom)
    # ============= QM COMPONENT ==============
    # first we need to compute iQM transformed to the iQM+Buf set
    a = zeros(geom.nX)
    a[geom["iQM"]] = 1
    a = a[geom["iQM+Buf"]]
    iQM_in_QMbuf = find(a)  # just geom["iQM"]!!!
    # now we can compute the sum of the site-gradients
    dE_qm = compute(Y[:, geom["iQM+Buf"]], "dEs", params; I=iQM_in_QMbuf)
    
    # ============= MM COMPONENT ==============
    # sum over the iMME list, which contains a layer of atoms in the
    # boundary region, in order to remove the boundary layer effect
    dE_mm = evalSiteFunction_D1(sp, geom, Y, geom["iMME"])
    # delete the forces in the boundary region (where atoms do not move)
    dE_mm[:, geom["iBC"]] = 0.0

    # ==========
    # put them together and return
    dE_mm[:, geom["iQM+Buf"]] += dE_qm
    return dE_mm
end
