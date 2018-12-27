# elapsed time: 0.74167516 seconds (456069752 bytes allocated, 20.26% gc time)

 
# function fermidist(epsilon, params)
# =====================================
# little helper function that computes the fermi distribution, the
# energy density, and its derivatives.
function fermidist(epsilon, params)
    E = exp(params.beta * (epsilon - params.eF))
    dE = params.beta*E
    hE = params.beta^2*E
    Wf = (1 + E).^(-1)
    Fermi = epsilon .* Wf
    dFermi = Wf - epsilon .* (1+E).^(-2) .* dE
    hFermi = -2*(1+E).^(-2) .* dE - epsilon .* ( -2*(1+E).^(-3) .* dE.^2 + (1+E).^(-2) .* hE )
    return Wf, Fermi, dFermi, hFermi
end



# function sorted_eig
# just a quick helper function to compute eigenvalues, then sort them
function sorted_eig(H)
    if any(isnan(H)) || any(isinf(H))
        error("sorted_eig : H contains NaNs or Infs!")
    end
    epsn, C = eig(Symmetric(H))
    Isort = sortperm(epsn)
    epsn = epsn[Isort]
    C = C[:,Isort]

    return epsn, C
end


# function compute(x, tasks, params; I = [])
# ==========================================
# energy and forces for tight-binding
#
# INPUT (all are compulsory)
#   x: positions, d x N
#   flags: list of indicators what is needed
#          E, dE, Es, dEs, rho, epsilon
#   params: TB.params list
#   I : indices that go into the site energy; this ignored for all tasks
#       except `Es`, `dEs`.
#
# Returns whatever is asked for in `tasks`. E.g.,
#   `E = totalE(x, 'E', params)`   returns the energy
#   `dE = totalE(x, 'G', params)`   returns the gradient  (or, "dE")
#   `E, dE = totalE(x, ('E', 'G'), params)`   returns both
#
# Hessian currently not implemented; use `totalE_fdhess` instead (TODO)
#
function compute(x::Array{Float64, 2},
                 tasks,
                 params;
                 I = Int[]::Array{Int, 1},
                 scalar_grad = false)
    
    # read input,
    d, N = size(x)
    
    # compute hamiltonian
    #  TODO: compute only what is needed
    H::Array{Float64,2},
    dH::Array{Float64,3},
    P::Array{Float64,2},
    dP::Array{Float64,3} = params.hamiltonian(x, ("H", "dH", "P", "dP"))

    # Compute the spectral decomposition and sort by eigenvalue magnitude
    epsilon::Array{Float64,1},
    C::Array{Float64,2} = sorted_eig(H)

    #  compute the energy and the density matrix
    Wf::Array{Float64,1},
    Fermi::Array{Float64,1},
    dFermi::Array{Float64,1} = fermidist(epsilon, params)

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
            E = sum(Fermi) + sum(P[:])
            ret = tuple(ret..., E)

        # ASSEMBLE THE FORCES
        # ====================
        elseif cur_task == "dE"
            # compute \epsilon_{s,i\alpha}
            #   > note that epsn_si should really be (d, nEps, nX)
            #   > This automatically becomes O(N) when the hamiltonian becomes sparse
            #   > The total forces then become O(N^2)
            #   > probably possible to reduce further with localisation ideas

            ## VERSION 1
            ## dEpsn = [ 2.0 * C[i,s] * dot(C[:, s], slice(dH, a, i, :))
            ##          for a=1:d, s=1:N, i=1:N]
            ## dE = [ (dot(dFermi, slice(dEpsn, a, :, i)) + 2*sum(dP[a, i, :][:]))
            ##       for a = 1:d, i=1:N]

            ## VERSION 2
            ## dE = zeros(d, N)
            ## for a=1:d, i=1:N
            ##     dEpsn_a_i =  [2.0 * C[i,s] * dot(C[:, s], slice(dH, a, i, :))
            ##                   for s=1:N]
            ##     dE[a,i] = dot(dFermi, dEpsn_a_i) + 2.0 * sum(slice(dP, a, i, :))
            ## end
            
            ## VERSION 3
            # dE = 2.0 * Float64[ ( sum(slice(dP, a, i, :))
            #            + dot( dFermi, slice(C,i,:) .* (C' * slice(dH,a,i,:)) ) )
            #                   for a = 1:d, i = 1:N ]
            
            dE = 2.0 * Float64[ ( sum(view(dP, a, i, :))
                        + dot( dFermi, view(C,i,:) .* (C' * view(dH, a, i, :)) ) )
                               for a = 1:d, i = 1:N ]
            if scalar_grad
                dE = dE[:]
            end
            
            # # test relative magnitude of forces test
            # dErep =  2.0 * Float64[ sum(slice(dP, a, i, :))
            #                         for a = 1:d, i = 1:N ]
            # dEbond = 2.0 * Float64[
            #             (dot( dFermi, slice(C,i,:) .* (C' * slice(dH,a,i,:)) ))
            #                        for a = 1:d, i = 1:N ]
            # @show norm(dErep[:], Inf)
            # @show norm(dEbond[:], Inf)
            
            # update returns list
            ret = tuple(ret..., dE)
            
        # ASSEMBLE THE DENSITY MATRIX
        # ===========================
        # TODO, this is not tested? Probably because I never actually use
        # the density matrix
        elseif lowercase(cur_task) == "rho"
            rho = zeros(N,N)
            for s = 1:N
                rho += Wf[s] * C[:, s] * C[:, s]'
            end
            ret = tuple(ret..., rho)
            
        # ENERGY LEVELS
        # ==============
        elseif lowercase(cur_task) == "epsilon"
            ret = tuple(ret..., epsilon)
            
        # SITE ENERGY
        # ============
        # If E = \sum_s f(\eps_s)
        #      = \sum_s f(\eps_s) \sum_{i\alpha} [c_s]_{i\alpha}^2,
        # then
        #    E_i = \sum_\alpha \sum_s f(\eps_s) [c_s]_{i\alpha}^2
        # (though in the current implementation there is only one orbital per
        # site, so the sum over \alpha is ignored
        #    For reference, the implementation of the total energy read
        #            E = sum(Fermi) + sum(P[:])
        elseif cur_task == "Es"
		#	typeof(I)
		#	println(typeof(I))
		#	typeof(C)
		#	println(typeof(C))
		#	typeof(Fermi)
		#	println(typeof(Fermi))
        #    Es = sum(C[I, :].^2 * Fermi) # + sum(P[I, :][:])
		#	println(typeof(I))
		#	println(C)
		#	println(I)
		#		println(typeof(C))
		# There may be further problem is I is an array and dot should be used here to leave the explicit invokation of sum.
            Es = sum(C[I, :].^2 .* Fermi') # + sum(P[I, :][:])
            # ============== DEBUG
            for i in I, j = 1:N
                #
                Es += P[i,j]
            end
            # DEBUG ==============
            ret = tuple(ret..., Es)
            
        # DERIVATIVE OF SITE ENERGY
        # =========================
        elseif cur_task == "dEs-old"
            # throw(ArgumentError("Tb.totalE: task `dEs` not yet implemented"))
            dEs = zeros(Float64, d, N)

            # allocate a temporary array that enables later computation
            # via pure matrix-matrix multiplication
            G = zeros(Float64, N, N)
            g = zeros(Float64, N)

            # outer loop over dimension index - a, and over energy-level s
            for a = 1:d, s = 1:N
                
                # compute the G array
                fill!(g, 0.0)
                for ig = 1:N
                    for j = 1:N
                        G[j, ig] = dH[a,j,ig] * C[j,s]
                        g[j] += dH[a,j,ig] * C[ig,s]
                    end
                end
                for j = 1:N
                    G[j,j] += g[j]
                end
                # compute the epsn_s_j > j-th entry of D_epsn_s
                D_epsn_s =  G * C[:, s]
				
                # the following line is probably the bottle-neck!
                # g = - (C' * g) ./ (epsilon - epsilon[s])
                G = - G * C
                for ig = 1:N
                    diff_eps = epsilon[ig]-epsilon[s]
                    if ig != s
                        for iii = 1:N
                            G[iii,ig] /= diff_eps
                        end
                    else
                        for ii = 1:N; G[ii,ig] = 0.0; end
                    end
                end
                
	        # invert coordinate transform  
                # csI_j = C[I,:] .* G'
                
                # add the computed values to the site energy derivative
                # >>> idealised O(1)
                #  NOTE the weird construction of [C[I, s]] is just to
                #    circumvent the strange situation that if I has just
                #    one entry, then C[I,s] is a scalar while  csI_j is
                #    an array, and `dot` does not like this
				# csI_j_1 = G*C[I,:]
                if length(I) == 1
					csI_j_1 = G*C[I,:]
					# Base.LinAlg.BLAS.gemv!('N', 1.0, G, C[I,:], 0.0, csI_j_1)
                else
                    csI_j = C[I,:] * G'
                end
				#    for j = 1:N
				#	println(C[I,s])
				#	println([C[I,s];])
				#	println(csI_j[:,j])
				#	println(typeof([C[I,s];]))
				#	println(typeof(csI_j[:,j]))
                #    dEs[a, j] += dFermi[s] * D_epsn_s[j] * sum( C[I,s].^2 ) +
                #                 2. * Fermi[s] * dot( [C[I,s];], csI_j[:, j] )
                #    dEs[a, j] += dFermi[s] * D_epsn_s[j] * sum( C[I,s].^2 ) + 2. * Fermi[s] .* dot( [C[I,s];], [csI_j[I, j];] )
				#	cI = C[I, :]
                    if length(I) == 1
                        for j = 1:N
                            dEs[a, j] += dFermi[s] * D_epsn_s[j] * C[I,s]^2 +
                            2. * Fermi[s] * C[I,s] * csI_j_1[j]
                        end
                    else
                        for j = 1:N
                            dEs[a, j] += dFermi[s] * D_epsn_s[j] * sum( C[I,s].^2 ) +
                            2. * Fermi[s] * dot( [C[I, s]], csI_j[:, j] )
                        end
                    end
                # end
            end

            # adding the pair-potential components to the site energy
            #  >>>  O(N * length(I))
            for a = 1:d, j = 1:N, i in I
                dEs[a,j] -= dP[a,i,j]
                dEs[a,i] += dP[a,i,j]
            end

            # update return tuple
            ret = tuple(ret..., dEs)

            
        # DERIVATIVE OF SITE ENERGY
        # =========================
        elseif cur_task == "dEs"
            # throw(ArgumentError("Tb.totalE: task `dEs` not yet implemented"))
            dEs = zeros(Float64, d, N)

            # allocate temporary arrays that enable later computation
            # via in-place BLAS
            G = zeros(Float64, N, N)
            G1 = zeros(Float64, N, N)
            G2 = zeros(Float64, N, N)
            g = zeros(Float64, N)
            dHa = zeros(Float64, N, N)
            c = zeros(Float64, N)
            diff_eps_inv = zeros(Float64, N)
            D_epsn_s = zeros(Float64, N)
            if length(I) == 1
			#	println(typeof(C[I, :]))
			#    cI = squeeze(C[I, :], 1) # no need squeeze since C[I,:] already ∈ Array{Float64,1}
				cI = C[I, :]
            else
                cI = C[I, :]
            end
            csI_j_1 = zeros(Float64, N)
            

            # outer loop over dimension index - a, and over energy-level s
            for a = 1:d
            #    dHa[:,:] = slice(dH, a, :, :)
				dHa[:,:] = view(dH, a, :, :)
                # a little precomputation - BLAS-ised for performance
                Base.LinAlg.BLAS.gemm!('N', 'N', 1.0, dHa, C, 0.0, G1)
                # gs[j] = dHa[j,:] . C[:,s]
                
                for s = 1:N
                    # copy a part of C that we need to use with BLAS
                    for iii = 1:N; c[iii] = C[iii,s]; end    
                
                    # compute the G array
                    for ig = 1:N
                        @simd for j=1:N
                            @inbounds G[j, ig] = dHa[j,ig] * c[j]
                        end
                    end
                    @inbounds for j = 1:N
                        G[j,j] += G1[j,s]
                    end
                    
                    # compute the epsn_s_j > j-th entry of D_epsn_s
                    # D_epsn_s =  G * c # C[:, s]
                    Base.LinAlg.BLAS.gemv!('N', 1.0, G, c, 0.0,  D_epsn_s)

                    # the following line was the bottle-neck at some point,
                    # after moving to BLAS, not anymore
                    # g = - (C' * g) ./ (epsilon - epsilon[s])
                    # G2 = - G * C
                    Base.LinAlg.BLAS.gemm!('N', 'N', -1.0, G, C, 0.0, G2)
                    # divide by energy
                    @inbounds for iii = 1:N
                        diff_eps_inv[iii] = 1/(epsilon[iii] - epsilon[s])
                    end

                    # NASTY HACK to fix the multiple e-val issue!
                    for iii = max(1,s-1):min(N, s+1)
                        if abs(diff_eps_inv[iii]) > 1e10
                            diff_eps_inv[iii] = 0.0
                        end
                    end

                    ## ##### DEBUG  ####
                    ## println(round(diff_eps_inv[max(1,s-5):min(N,s+5)], 2))
                    
                    for ig = 1:N
                        @simd for iii = 1:N
                            @inbounds G2[iii,ig] *= diff_eps_inv[ig]
                        end
                    end
                
	            # invert coordinate transform
                    if length(I) == 1
                        Base.LinAlg.BLAS.gemv!('N', 1.0, G2, cI, 0.0, csI_j_1)
                    else
                        csI_j = C[I,:] * G2'

                    end
                
                    # add the computed values to the site energy derivative
                    # >>> idealised O(1)
                    #  NOTE the weird construction of [C[I, s]] is just to
                    #    circumvent the strange situation that if I has just
                    #    one entry, then C[I,s] is a scalar while  csI_j is
                    #    an array, and `dot` does not like this
                    if length(I) == 1
                        for j = 1:N
                            dEs[a, j] += dFermi[s] * D_epsn_s[j] * cI[s]^2 +
                            2. * Fermi[s] * cI[s] * csI_j_1[j]
                        end
                    else
                        for j = 1:N
                            dEs[a, j] += dFermi[s] * D_epsn_s[j] * sum( C[I,s].^2 ) +
                            2. * Fermi[s] * dot( C[I, s], csI_j[:, j] )
                        end
                    end
                end # for s
            end # for a
            
            # adding the pair-potential components to the site energy
            #  >>>  O(N * length(I))
            for a = 1:d, j = 1:N, i in I
                dEs[a,j] -= dP[a,i,j]
                dEs[a,i] += dP[a,i,j]
            end

            # update return tuple
            ret = tuple(ret..., dEs)
            

            
        # UNKNOWN TASK > through exception
        else
            throw(ArgumentError("TB.compute: unknown task"))

        end # if task[id] ==
    end # for id; end of loop through tasks

    if length(ret) == 1
        return ret[1]
    else
        return ret
    end
end # end of `function totalE`




## function totalE_fd_hess(x, params; i0)
##========================================
# finite-difference hessian approximation to totalE
# Parameters:
#   x : positions (d x N)
#   params : tbparams structure
#   i0 : if i0==[] (default) then the complete hessian is computed
#          (this is not yet implemented >>> TODO)
#        if i0 is an integer, then only the i0-row is computed
#
#  Returns
#    hE : d x N x d x N matrix if i0 == []
#         or d x d x N if i0 is an integer
#
function totalE_fd_hess(x, params; i0 = [])
    d, N = size(x)
    if isempty(i0)
        throw(ArgumentError("TB.totalE_fd_hess : total hessian to be yet implemented"))
    else
        # Compute the i0-th row using a second-order centered finite difference
        Hi0 = zeros(d, d, N)
        for a = 1:d
            x[a, i0] += params.h
            dEp = compute(x, "dE", params)
            x[a, i0] -= 2*params.h
            dEm = compute(x, "dE", params)
            Hi0[a, :, :] = (dEp-dEm) / (2*params.h)
            x[a, i0] += params.h
        end
    end
    return Hi0
end


## function siteE_fd_hess(x, params, i0)
##========================================
# finite-difference hessian approximation to E_{i0}
# Parameters:
#   x : positions (d x N)
#   params : tbparams structure
#   i0 : site for which the site energy hessian is to be computed
#
#  Returns
#    hEs : d x N x d x N matrix
#
function siteE_fd_hess(x, params, i0)
    d, N = size(x)
    # allocate space
    hEs = zeros(d, N, d, N)
    # loop through one set of (a,j) ~ dimension x nodes
    for a = 1:d, j = 1:N
        x[a,j] += params.h
    #    dEs_p = compute(x, "dEs", params; I=[i0;])
        dEs_p = compute(x, "dEs", params; I=i0)
        x[a,j] -= 2 * params.h
    #    dEs_m = compute(x, "dEs", params; I=[i0;])
        dEs_m = compute(x, "dEs", params; I=i0)
        hEs[a, j, :, :] = (dEs_p - dEs_m) / (2*params.h)
        x[a,j] += params.h
    end
    return hEs
end

## function siteE_fd_hess_fd(x, params, i0)
##========================================
# Use double finite-difference to 3-order derivative approximation to E_{i0}
# Parameters:
#   x : positions (d x N)
#   params : tbparams structure
#   i0 : site for which the site energy hessian is to be computed
#
#  Returns
#    fdhEs : d x N x d x N x d x N matrix 
#
function siteE_fd_hess_fd(x, params, i0)
    d, N = size(x)
    # allocate space
    fdhEs = zeros(d, N, d, N, d, N)
    # loop through one set of (a,j) ~ dimension x nodes
    for a = 1:d, j = 1:N
        x[a, j] += params.h
    #    dEs_p = compute(x, "dEs", params; I=[i0;])
        hEs_p = siteE_fd_hess(x, params, i0)
        x[a, j] -= 2 * params.h
    #    dEs_m = compute(x, "dEs", params; I=[i0;])
        hEs_m = siteE_fd_hess(x, params, i0)
        fdhEs[a, j, :, :, :, :] = (hEs_p - hEs_m) / (2*params.h)
        x[a, j] += params.h
    end
    return fdhEs
end


# function lattice_energy(F, params; nBox = 3)
# ========================================
#    This is a dumb implementation of the Cauchy-Born energy density
#    using the site energy functional.
#
#  INPUT:
#    F : d x d matrix  (deformation gradient)
#    params : a tbparams structure
#    F0 : the reference crystal is given by F0 Z^d
#    radius : radius of the computational region in which the site energy
#             is evaluated.
#
#  OUTPUT: (W, dW)
#    stored energy density and 1st Piola Kirchhoff stress
#
function lattice_energy(F, params; F0 = [], radius = 5.0)
    # problem dimension
    d = size(F, 2)
    # create suitable F0, if none was defined
    if F0 == []
        F0 = eye(d)
    end
    
    # compute a reasonable neighbourhood of the origin
    # taking into account the site energy decay
    # 5 is a bit conservative maybe.
    xF0, I0 = generate_AZd_ball(F0, radius+1e-10)
    
    # Evaluate the site energy and gradient at the origin
    # Es, dEs = compute(F * xF0, ("Es", "dEs"), params; I = [I0])
	if length(I0) == 1
	    Es, dEs = compute(F * xF0, ("Es", "dEs"), params; I = I0[1])
    else
	    Es, dEs = compute(F * xF0, ("Es", "dEs"), params; I = I0)
	end
    # Compute the CB energy and stress from this
    W = Es
    dW = zeros(d,d)
    for i = 1:size(xF0, 2)
        rho = xF0[:,i] - xF0[:,I0]
        dW += dEs[:,i] * rho'
    end
    
    # apply the volume scaling (relative to F0)
    W *= det(F0)
    dW *= det(F0)
    
    return W, dW
end
