




#
# sequential QM, QM/MM-frc, QM/MM-en solver for benchmarking, to
# save copmutational time (and to ensure all three converge to the
# same local minimiser
#
# WARNING: this is a quick-and-dirty code that needs to be tossed out eventually
#
# We do not need Y0 = Nothing
function solve_all(params::tbparams, geom::tbgeom;
                   disp = 1,
                   Y0 = [], sp_R = 0, sp_Rbuf = 0,
                   reference = false)

    
    # ============ STEP 1: PURE QM SOLVE OF CORE REGION ==============
    if disp >= 1
        println("Step 1: pure QM solve of the core region"); sleep(0.01)
    end
    Yq, Eq, α_hist = solve("QM", params, geom, Y0=Y0, disp=disp, Ediff=true)
    
    
    # ============ SOME PREPS FOR STEPS 1 AND 2 ==================
    if disp >= 1
        println("------------------------------------------")
        println(" preparations for QM/MM");
    end
    # compute the set of free indices
    iFreeX = sort(union(geom["iQM+Buf"], geom["iMM"]))
    # construct the free DOFs
    a = zeros(2, geom.nX); a[:, iFreeX] = 1; a = a[:]
    iFreeDOFs = find(a)

    if disp >= 1
        println("Construct Preconditioner"); sleep(0.01)
    end
    M = 20. * prec_laplace(geom, iFreeX)

    # without further information, the initial guess should be the
    # reference configuration
	if isempty(Y0)
        Y0 = geom.X
    end

    # guess a good alpha; mean / 2 seemed a bit conservative, so we try
    # something more aggressive
    # α = mean(α_hist) / 2 >>> ~ 1.2 (73 it) , 1.5 (54 its),
    #    1.4 is a safe compromise?
    # α = 1.0
    α = mean(α_hist) / 2

    # ============ STEP 2: QM/MM-FORCE SOLVER  ==============
    if disp >= 1
        println("------------------------------------------")
        println("Step 2: static QM/MM-frc solve"); sleep(0.01)
    end

    # construct the site force
    if disp >= 1
        println("Construct Site Force"); sleep(0.01)
    end
    # if no radii are passed, then take the following default (error for now)
    if (sp_R == 0) || (sp_Rbuf == 0)
        error("must pass valid sp_R, sp_Rbuf")
    end
    frcMM = TB.siteForceApproximation(params, geom["A"], sp_R, sp_Rbuf)

    # create a force wrapper
    function Frc_Wrapper(y)
        Y1 = copy(Y0)
        Y1[iFreeDOFs] = y
        F1 = QM_MM_frc(Y1, "F", frcMM, params, geom)
        return  F1[iFreeDOFs]
    end
    
    # start the static solver, with the QM solution as the starting guess
    y, F = optim_Static(Frc_Wrapper, Yq[iFreeDOFs], M, α;
                        disp = disp, xtol = 1e-8, gtol = 1e-8, minnit = 10)
    
    # reconstruct the solution to return
    Yf = copy(Y0)
    Yf[iFreeDOFs] = y            


    # otherwise compute QM/MM-energy next
    # ============ STEP 3: PURE QM SOLVE OF CORE REGION ==============
    if disp >= 1
        println("------------------------------------------")
        println("Step 3: static QM/MM-E solve"); sleep(0.01)
    end

    # construct the site force
    if disp >= 1
        println("Construct Site Energy"); sleep(0.01)
    end
    tic()
    enMM = TB.constructSiteQuadraticE(params, geom["A"], sp_R, sp_Rbuf)
    toc()
    
    # if we are meant to just compute the reference solution, then
    # return the QM/MM-frc solution together with the hybrid energy
    #   (see how that goes?!?!?)
    if reference
        println("""This is the reference computation, so we don't need to run
                   the QM/MM-E scheme; we only evaluate one reference energy!""")
        Ee = QM_MM_en_E(Yf, enMM, params, geom)
        Ee0 = QM_MM_en_E(geom.X, enMM, params, geom)
        return Yf, (Ee-Ee0)
    end

    # create an energy wrapper
    function Frc_Wrapper_E(y)
        Y = copy(Y0)
        Y[iFreeDOFs] = y
        F = QM_MM_en_dE(Y, enMM, params, geom)
        return F[iFreeDOFs]
    end
    
    # start the static solver, with the QM solution as the starting guess
    y, F = optim_Static(Frc_Wrapper_E, Yf[iFreeDOFs], M, α;
                        disp=disp, xtol=1e-8, gtol = 1e-8, minnit=10)
    
    # reconstruct the solution to return
    Ye = copy(Y0)
    Ye[iFreeDOFs] = y
    
    # compute the hybrid energy?
    Ee = QM_MM_en_E(Ye, enMM, params, geom)
    Ee0 = QM_MM_en_E(geom.X, enMM, params, geom)
    
    return Yq, Yf, Ye, Eq, Ee-Ee0
    #return Yq, Yf, Ye
end



