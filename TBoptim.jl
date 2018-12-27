


# function prec_laplace(geom::tbgeom)
#    a cheap, basic Laplace-type preconditioner
function prec_laplace(geom::tbgeom)

    # nearest neighbour directions, this is a dim x N array
    # there N is the number of "nearest neighbours"
    Rg_nn = getNearestNeigRg(geom)

    # allocate some space
    ntrip_est = geom.nX * (size(Rg_nn, 2) + 1)
    A = SparseTriplet(ntrip_est)

    # loop through all "live" sites
    for n = 1:geom.nX
        # collect the nearest neighbours
        I_nn = getNeighbourhood(geom, n, Rg_nn)
        # create matrix entries for neighbours
        n_neigs = 0.0
        for i in I_nn
            if i != 0
                n_neigs += 1.0
                A[n, i] += -1.0
            end
        end
        # diagonal entry
        if n_neigs == 0
            print("*")
        end
        A[n,n] += n_neigs
    end

    # return the sparse matrix
    Asp = sparse(A)
    return kron(Asp, speye(geom.dim))  # change from 2 to 3
end


# function prec_laplace(geom::tbgeom, iFreeX)
#    a cheap, basic Laplace-type preconditioner
function prec_laplace(geom::tbgeom, iFreeX)

    # construct dof numbering
    dof_idx = zeros(geom.nX)
    for n = 1:length(iFreeX)
        dof_idx[iFreeX[n]] = n
    end
    
    # nearest neighbour directions, this is a dim x N array
    # there N is the number of "nearest neighbours"
    Rg_nn = getNearestNeigRg(geom)
    #println(Rg_nn)

    # allocate some space
    ntrip_est = geom.nX * (size(Rg_nn, 2) + 1)
    A = SparseTriplet(ntrip_est)

    # loop through all "live" sites
    for n in iFreeX
        # collect the nearest neighbours
        I_nn = getNeighbourhood(geom, n, Rg_nn)
        # create matrix entries for neighbours
        n_neigs = 0.0
        for i in I_nn
            if i != 0
                n_neigs += 1.0
                if dof_idx[i] != 0
                    A[dof_idx[n], dof_idx[i]] += -1.0
                end
            end
        end
        # diagonal entry
        if n_neigs == 0
            print("*")
        end
        A[dof_idx[n],dof_idx[n]] += n_neigs
    end

    # return the sparse matrix
    Asp = sparse(A)

    # Bsp = kron(Asp, speye(geom.dim))
    # a = zeros(2, geom.nX); a[:, iFreeX] = 1; a = a[:]
    # iFreeDOFs = find(a)
    # Bsp = Bsp[iFreeDOFs, iFreeDOFs]
    # return Bsp

    # Asp = Asp[iFreeX, iFreeX]
    return  kron(Asp, speye(geom.dim))  # change from 2 to 3 
end


# function prec_laplace_screw(geom::tbgeom, iFreeX)
#    a cheap, basic Laplace-type preconditioner for screw dislocation
# screw dislocation x, y fixed
function prec_laplace_screw(geom::tbgeom, iFreeX)

    # construct dof numbering
    dof_idx = zeros(geom.nX)
    for n = 1:length(iFreeX)
        dof_idx[iFreeX[n]] = n
    end
    
    # nearest neighbour directions, this is a dim x N array
    # there N is the number of "nearest neighbours"
    Rg_nn = getNearestNeigRg(geom)
    #println(Rg_nn)

    # allocate some space
    ntrip_est = geom.nX * (size(Rg_nn, 2) + 1)
    A = SparseTriplet(ntrip_est)

    # loop through all "live" sites
    for n in iFreeX
        # collect the nearest neighbours
        I_nn = getNeighbourhood(geom, n, Rg_nn)
        # create matrix entries for neighbours
        n_neigs = 0.0
        for i in I_nn
            if i != 0
                n_neigs += 1.0
                if dof_idx[i] != 0
                    A[dof_idx[n], dof_idx[i]] += -1.0
                end
            end
        end
        # diagonal entry
        if n_neigs == 0
            print("*")
        end
        A[dof_idx[n],dof_idx[n]] += n_neigs
    end

    # return the sparse matrix
    Asp = sparse(A)

    return  kron(Asp, speye(geom.dim)+1)  # change from 2 to 3 
end




# function optim_SimpleSD(Efun, Y, M;
#                         gtol = 1e-6, xtol = 1e-6, maxnit = 200 )
#
#   Y : d x nX array of points, or more general dofs
#   geom : tbgeom object
#   M : preconditioner
#   -------------------
#   tolerance settings
#
function optim_SimpleSD(Efun, x, M;
                        gtol = 1e-6, xtol = 1e-6, maxnit = 200,
                        Carmijo = 0.1, α_min = 1e-8, disp = 1 )

    # initialise the preconditioner
    # @show M
    # M = full(M)
    # println(M == M')
    cholfactM = cholfact(M)

    if disp ≥ 2
        @printf(" nit   |   ΔE        |∇E|∞     |Δx|∞      α \n")
        @printf("-------|----------------------------------------\n")
        sleep(0.01)
    end

    # allocate and initialise some variables that we need to
    # read outside the look as well
    xres = xtol + 1
    E = 0; ∇E = []; Eold = E
    error = 0
    nit = 0
    α_hist = zeros(5)
	α = 0
	p_dot_∇E_old = 0
    # step right into the optimisation loop
    for nit = 1:maxnit
        # evaluate energy and forces
        # we are doing some multiple evaluations here, but in reality
        # the energy is a very cheap extra.
        E, ∇E = Efun(x, ("E", "dE"))
        if any(isnan(∇E)) || any(isinf(∇E))
            error("optim_SimpleSD: dE contains NaNs or Infs")
        end
        
        # apply preconditioner, compute residuals
        p = - (cholfactM \ ∇E)
        p_dot_∇E = dot(p, ∇E)
        gres = norm(∇E, Inf)

        # make an initial guess for the linesearch
        if nit > 1
            α *= max(1/4, min(2, p_dot_∇E_old / p_dot_∇E))
		#	println(α)
        else
            α = 1.0
        end
        # store α history
        α_hist = [α; α_hist[1:4]]

        # output iteration info
        if disp >= 2
            @printf(" %5d | %4.2e  %4.2e  %4.2e  %4.2e \n",
                    nit, E - Eold, gres, xres, α); sleep(0.01)
        end

        # check for termination
        if (  (xres ≤ xtol) && ((gres ≤ gtol) || (p_dot_∇E < 1e-12)) )
            error = -1
            break
        end

        # linesearch
        while Efun(x+α*p, "E") ≥ E + α * Carmijo * p_dot_∇E
            α *= 0.5
            if α < α_min
                break
            end
        end
        if α < α_min
            error = 1
            break
        end

        # updates
        xres = α * norm(p, Inf)
        x +=  α * p
        Eold = E
        p_dot_∇E_old = p_dot_∇E
    end

    # output termination message
    if disp >= 1
        if error == -1
            println("optim_SimpleSD terminated succesfully in ",
                    nit, " iterations")
        elseif error == 1
            println("optim_SimpleSD terminated because α < α_min")
        else
            println("optim_SimpleSD terminated because nit > maxnit")
        end
        sleep(0.01)
    end

    # return some useful data
    return x, E, ∇E, α_hist
end





# function optim_SimpleSD(Efun, Y, M;
#                         gtol = 1e-6, xtol = 1e-6, maxnit = 200 )
#
#   Y : d x nX array of points, or more general dofs
#   geom : tbgeom object
#   M : preconditioner
#   -------------------
#   tolerance settings
#
function optim_Static(FrcFun, x, M, α;
                        gtol = 1e-7, xtol = 1e-7, maxnit = 500,
                        disp = 1, minnit = 0 )

    # initialise the preconditioner
    cholfactM = cholfact(M)

    if disp ≥ 2
        @printf(" α = %4.2e \n", α)
        @printf("-------|---------------------\n")
        @printf(" nit   |   |F|∞     |Δx|∞/α  \n")
        @printf("-------|---------------------\n"); sleep(0.01)
    end

    # allocate and initialise some variables that we need to
    # read outside the look as well
    xres = xtol + 1
    F = []
    error = 0
    nit = 0
    # step right into the optimisation loop
    for nit = 1:maxnit
        # evaluate energy and forces
        # we are doing some multiple evaluations here, but in reality
        # the energy is a very cheap extra.
        F = FrcFun(x)
        
        # apply preconditioner, compute residuals
        p = - (cholfactM \ F)
        
        # save residuals
        gres = norm(F, Inf)
        xres = norm(p, Inf)
        
        # output iteration info
        if disp >= 2
            @printf(" %5d | %4.2e  %4.2e  \n",
                    nit, gres, xres); sleep(0.01)
        end

        # check for termination
        if nit > minnit
            if (  (xres ≤ xtol) && (gres ≤ gtol) )
                error = -1
                break
            end
        end

        # updates
        x +=  α * p
    end

    
    # output termination message
    if disp >= 1
        if error == -1
            println("optim_Static terminated succesfully in ",
                    nit, " iterations")
        else
            println("optim_Static terminated because nit > maxnit")
        end
        sleep(0.01)
    end

    # return some useful data
    return x, F
end





@doc """
function solve(model, params::tbparams, geom::tbgeom)
    solver wrapper
""" ->
function solve(model, params, geom::tbgeom;
			   Y0 = [], disp = 1, maxnit = 1000, Ediff=false)

    if model != "QM"
        error("""TB.solve : only model = "QM" is currently allowed""")
    end

    # TODO: if geom["iFree"] does not exist, then extract it from the
    #       geometry information
    iFreeX = sort(geom["iQM"])
    # construct the free DOFs
    a = zeros(2, geom.nX); a[:, iFreeX] = 1; a = a[:]
    iFreeDOFs = find(a)

    # create a preconditioner
    println("Construct Preconditioner"); sleep(0.01)
    # M = 20. * prec_laplace(geom)
    # M = M[iFreeDOFs, iFreeDOFs]
    
    M = 20. * prec_laplace(geom, iFreeX)
    #M = speye(273, 273)


    # create a wrapper for the energy
    function Ewrapper(x, tasks, params, geom, iFreeX)
        Y = copy(geom.X)
        Y[:, iFreeX] = reshape(x, (2, length(iFreeX)))  # change from 2 to 3
        ret = TB.compute(Y[:, geom["iQM+Buf"]], tasks, params)
        if length(ret) == 1  # energy only
            return ret[1]
        else                 # energy and forces
            E = ret[1]
            ∇E = zeros(geom.dim, geom.nX)  # change from 2 to 3
            ∇E[:, geom["iQM+Buf"]] = ret[2]
            ∇E = ∇E[:, iFreeX]
            #∇E[1:2, :] = 0.0  
            return E, ∇E[:]
        end
    end
    
    # without further information, the initial guess should be the
    # reference configuration
	if isempty(Y0) 
        Y0 = geom.X
    end

    # specify the objective function
    Eobj = (x, tasks) -> Ewrapper(x, tasks, params, geom, iFreeX)

    # start the optimiser : TODO: allow choices
    y, E, ∇E, α_hist = optim_SimpleSD(Eobj, (Y0[:, iFreeX])[:], M;
                                      maxnit = maxnit, disp = disp)

    # reconstruct the solution to return
    Y = copy(geom.X)
    Y[:, iFreeX] = reshape(y, (2, length(iFreeX)))  

    # TODO: if we are also returning ∇E, then it is not entirely clear which
    #       format to return it in.

    # compute energy-difference to starting configuration
    if Ediff
        E0 = TB.compute(Y0[:, geom["iQM+Buf"]], "E", params)
            # Eobj((Y0[:, iFreeX])[:], ("E"))
        E -= E0
    end

    # return
    return Y, E, α_hist
end



@doc """
function solve_screw(model, params::tbparams, geom::tbgeom)
    solver wrapper for screw dislocation
""" ->
function solve_screw(model, params, geom::tbgeom;
               Y0 = [], disp = 1, maxnit = 1000, Ediff=false)

    if model != "QM"
        error("""TB.solve_screw : only model = "QM" is currently allowed""")
    end

    # TODO: if geom["iFree"] does not exist, then extract it from the
    #       geometry information
    iFreeX = sort(geom["iQM"])
    # construct the free DOFs
    a = zeros(2, geom.nX); a[:, iFreeX] = 1; a = a[:]
    iFreeDOFs = find(a)

    # create a preconditioner
    println("Construct Preconditioner"); sleep(0.01)
    # M = 20. * prec_laplace(geom)
    # M = M[iFreeDOFs, iFreeDOFs]
    
    M = 20. * prec_laplace_screw(geom, iFreeX)
    #M = speye(273, 273)


    # create a wrapper for the energy
    function Ewrapper(x, tasks, params, geom, iFreeX)
        Y = copy(geom.X)
        Y[:, iFreeX] = reshape(x, (3, length(iFreeX)))  # change from 2 to 3
        ret = TB.compute(Y[:, geom["iQM+Buf"]], tasks, params)
        if length(ret) == 1  # energy only
            return ret[1]
        else                 # energy and forces
            E = ret[1]
            ∇E = zeros(geom.dim+1, geom.nX)  # change from 2 to 3
            ∇E[:, geom["iQM+Buf"]] = ret[2]
            ∇E = ∇E[:, iFreeX]
            ∇E[1:2, :] = 0.0  # x, y direction fixed
            return E, ∇E[:]
        end
    end
    
    if isempty(Y0) 
        Y0 = geom.X
    end

    # specify the objective function
    # optim_SimpleSD calculates the ("E", "dE")
    Eobj = (x, tasks) -> Ewrapper(x, tasks, params, geom, iFreeX)

    # start the optimiser : TODO: allow choices
    # screw do not change the minimizer
    y, E, ∇E, α_hist = optim_SimpleSD(Eobj, (Y0[:, iFreeX])[:], M;
                                      maxnit = maxnit, disp = disp)

    # reconstruct the solution to return
    Y = copy(geom.X)
    Y[:, iFreeX] = reshape(y, (2, length(iFreeX)))  

    # TODO: if we are also returning ∇E, then it is not entirely clear which
    #       format to return it in.

    # compute energy-difference to starting configuration
    if Ediff
        E0 = TB.compute(Y0[:, geom["iQM+Buf"]], "E", params)
            # Eobj((Y0[:, iFreeX])[:], ("E"))
        E -= E0
    end

    # return
    return Y, E, α_hist
end




# TODO: combine solve_QM_MM_frc with solve
function solve_QM_MM_frc(params::tbparams, geom::tbgeom;
						 Y0 = [], disp = 1, sp = [])

    # first do a pure QM solve for the core region only
    if disp >= 2
        println("First pass: pure QM solve of the core region");
        sleep(0.1)
    end
    Y, E, α_hist = solve("QM", params, geom, Y0=Y0, disp=disp)
    if disp >= 2
        println("------------------------------------------")
        println("Second pass: combined QM/MM solve")
        sleep(0.1)
    end

    # compute the set of free indices
    # TODO: if geom["iFree"] exists, then this needs to be used here instead
    #       since that will allow the user to impose additional kinematic
    #       constraints
    iFreeX = sort(union(geom["iQM+Buf"], geom["iMM"]))

    # construct the free DOFs
    a = zeros(2, geom.nX); a[:, iFreeX] = 1; a = a[:]
    iFreeDOFs = find(a)

    # # create a preconditioner
    # M = 20. * prec_laplace(geom)
    # # HERE IS THE BOTTLENECK!!!!!
    # M = M[iFreeDOFs, iFreeDOFs]

    println("Construct Preconditioner"); sleep(0.01)
    M = 20. * prec_laplace(geom, iFreeX)


    # without further information, the initial guess should be the
    # reference configuration
	if isempty(Y0)
        Y0 = geom.X
    end

    # construct the MM force
#	if isempty(sp)
	if ~isa(sp, SiteLinear)
        error("solve_QM_MM_frc: for the moment, the user must construct sp")
    end

    

    # specify the (negative) force function
    function Frc_Wrapper(y, sp, params, geom, iFreeDOFs, Y0)
        Y = copy(Y0)
        Y[iFreeDOFs] = y
        F = QM_MM_frc(Y, "F", sp, params, geom)
        return  F[iFreeDOFs]
    end
 
    # create the final force operator
    FrcFun = x -> Frc_Wrapper(x, sp, params, geom, iFreeDOFs, Y0)

    # guess a good alpha; this is a bit conservative but at most
    # we double the number of iterations.
    α = mean(α_hist) / 2
    
    # start the solver
    y, F = optim_Static(FrcFun, Y[iFreeDOFs], M, α; disp=disp)
    
    # reconstruct the solution to return
    Y = copy(geom.X)
    Y[iFreeDOFs] = y            

    # TODO: if we are also returning ∇E, then it is not entirely clear which
    #       format to return it in.
    return Y
end

