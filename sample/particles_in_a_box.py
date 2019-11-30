"""
    A 2-D simulation with 50 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution.
"""

from nbody import rand, animate, save

# Random particles filename
filename = "particles_in_a_box"

# Initializing a System() object via the rand() function
L = rand(100, x0 = (0,10), v0 = (0,1E4), q = (0,1E-2), m = (10,1), r = (1,0.2))

T = 5E-2
dt = 1E-4

# Solving for the given T and dt
L.solve(T, dt, collision = True, bounds = [[-50,50],[-50,50]])

# Saving the results to file
save(L, filename)

# Saving an animation of the system
animate(L)#, filename)
