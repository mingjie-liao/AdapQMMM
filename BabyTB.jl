
module BabyTB

###############################################################################
###  Implementation of a Baby Tight Binding Model
###############################################################################

# use some stuff from the tight binding module
using TB


# deformation matrix for triangular lattice
# const Atri = [[1.0 cos(π/3)], [0.0 sin(π/3)]]
const Atri = [1.0 cos(π/3); 0.0 sin(π/3)]


# # Constructor for `tbparams` object using the "babyTB model"
# tbparams() = TB.tbparams(1e-6, 50.0, 0.0, hamiltonian)

# Constructor for `tbparams` that accepts keyword arguments, with
# a simple default setting
# the default value for e0 is chosen so that the triangular lattice
# becomes an (approximate) equilibrium state / ground state??
# ML: Since we were 'using' TB, tbparams might not be extended.. here, tbparams differs from TB.tbparams!!
function tbparams(; alpha=4.0, etb = 0.5,
                  gamma = 4.0, e0 = 0.1,  S0 = 1.086, 
                  h = 1e-6, beta = 1.0, eF = 0.0 )
    myhamiltonian = (x, tasks) -> hamiltonian(x, tasks; alpha=alpha, etb=etb,
                                              gamma=gamma, e0=e0, S0=S0)
    return TB.tbparams(h, beta, eF, myhamiltonian)
end


# hamiltonian
# ============
# Compute the objects needed for the BabyTB model with Hamiltonian matrix
#    H_ij = exp(-alpha*rij), i \neq j
#         = 0, i = j
# and repulsive pair potential
#    phi(r) = e0 * exp(- gamma r )
# Parameters:
#       x : positions, d x N array
#   tasks : may be { "H", "dH", "hH", "P", "dP" }
#   alpha, gamma, e0 : model parameters; cf above.
#
function hamiltonian( x, tasks;
                      alpha=4.0, etb = 0.5, 
                      gamma = 4.0, e0 = 0.1, S0 = 1.09 )
    # extract dimensions
    d, N = size(x)
    # compute generic arrays distances
    R, S = TB.distance_matrices(x, 0)
    E = etb * exp(-alpha*(S-1)) .* sign(S)
    # Ep = e0 * exp(-gamma*S) .* sign(S)
    # dEp = (-gamma) * Ep
    Ep = e0 * (exp(-2*gamma*(S-S0)) - 2 * exp(-gamma*(S-S0))) .* sign(S)
    dEp = -2*gamma*e0*(exp(-2*gamma*(S-S0))-exp(-gamma*(S-S0))) .* sign(S)
    # assemble whatever is requested
    # first write tasks into a list (if it isn't already)
    if (typeof(tasks) == String) || (typeof(tasks)==Char)
        tasks = (tasks,)
    end
    # and create a returns list
    ret = ()
    for id = 1:length(tasks)
        ## H : HAMILTONIAN
        if tasks[id] == "H"
            ret = tuple(ret...,E)
        ## dH : Derivative of hamiltonian
        elseif tasks[id] == "dH"
            dH = Float64[ -alpha * E[n,m] * R[a,n,m] / (S[n,m]+eps())
                         for a = 1:d, n = 1:N, m=1:N ]
            ret = tuple(ret...,dH)
        ## hH : Second derivative of hamiltonian
        elseif tasks[id] == "hH"    #  NOT TESTED
            const del = eye(d)
            hH = [- alpha * E[m,n]*del[a,b] + alpha^2 * E[m,n]*R[a,m,n]*R[b,m,n]
                         for a=1:d, b=1:d, m=1:N, n=1:N ]
            ret = tuple(ret...,hH)
        ## P : Pair potential
        elseif tasks[id] == "P"
            ret = tuple(ret...,Ep)
        ## dP : Derivative of pair potential
        elseif tasks[id] == "dP"
            # NOTE: remember that dP_:ij = force of Pij bond acting onto i
            # dP = [ - gamma*Ep[m,n]*R[a,m,n]/(S[m,n]+eps())
            #       for a = 1:d, m = 1:N, n = 1:N ]
            dP = [ dEp[m,n]*R[a,m,n]/(S[m,n]+eps())
                  for a = 1:d, m = 1:N, n = 1:N ]
            ret = tuple(ret...,dP)
        else 
            throw(ArgumentError("tb_H_exp: illegal `tasks` argument"))
        end

    end # for id = 1:length(tasks)

    # return the constructed tuple
    # (or its first and only element, if only one is requested)
    if length(tasks) == 1
        return ret[1]
    else
        return ret
    end
    
end # function hamiltonian

         
end  # module BabyTB



