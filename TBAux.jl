

## Include auxiliary files and libraries
# meshgrid
include("Misc.jl")   




## tbparams
# =========
#   type that collects all the model and numerical parameters
#   h : finite-difference steplength
#   beta : temperature (parameter for Fermi distribution)
#   eF : Fermi level (parameters for Fermi distribution)
#   hamiltonian : function that returns the hamiltonian matrix or its
#                 derivatives, as a function of atom positions with
#                 surrounding vacuum.
type tbparams
    h::Float64
    beta::Float64
    eF::Float64
    hamiltonian::Function
end



# default constructor: write only garbage
# tbparams() = tbparams(0.0, 0.0, 0.0, none)
# ML: "none" is eliminated, "nothing" and "Nullable{Function}()" cannot be converted to type of Function. Use identity mapping to initial now...
tbparams() = tbparams(0.0, 0.0, 0.0, x->x)




## function generate_AZd_ball(A, R)
##==================================
# Input:
#    A : d x d
#    R : scalar
# Output:
#    x : d x ?; array of points inside the closed ball B_R \cap (A Z^d)
#
function generate_AZd_ball(A, R)
    # problem dimension
    d = size(A)[1]
    # get smallest singular value
    sig = minimum(svd(A)[2])
    # generate a cubic portion of Zd containing -R/sig : R/sig in each
    # coordinate direction
    ndim = ceil(R/sig)
    z = dgrid_list(-ndim:ndim, d)
    # transform via A
    x = A*z
    # find points inside the ball and return them
    I_ball = find( sumabs2(x, 1) .< R^2 )
    y = x[:, I_ball]
    I0 = find( sumabs(y, 1) .== 0.0 )
    return y, I0
end



## test_config(n, d, r)
## =======================
# A little routine to create a reasonable configuration to work with
#    n : how many atoms per dimension
#    d : space dimension
#    r : amount by which to perturb positions from Zd lattice
function test_config(n, d, r = 0.0)
    if d == 1
        x = 1:n
        x = x + r * (rand(n)-0.5)
        x = x[:]'
        return x
    elseif d == 2
        x = 1:n
        x, y = meshgrid(x, x)
        x = [x[:]'; y[:]']
        x += r * (rand(2, n^2)-0.5)
        x = [1 cos(pi/3); 0 sin(pi/3)] * x
        return x
    elseif d == 3
        x = 1:n
        x, y, z = meshgrid(x, x, x)
        x = [x[:]', y[:]', z[:]']
        x += r * (rand(3, n^3)-0.5)
        return x
    end
end




## distance_matrices(x, L)
## ========================
# R_:ij = x_i - x_j
# S_ij  = |x_i - x_j|
#
function distance_matrices(x, L)
    d, N = size(x)
    # compute vector distances
    R = Float64[ x[a, i] - x[a, j] for a=1:d, i=1:N, j=1:N]
    # impose periodic boundary conditions
    if L != 0; R = mod(R+L/2, L) - L/2; end
    # compute scalar distances 
    S = sqrt(squeeze(sumabs2(R, 1), 1)) # In-plane
    # S = sqrt(squeeze(sumabs2(R[3, :, :], 1), 1)) # Anti-plane

    return R, S
end







    

