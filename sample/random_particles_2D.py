"""
    A 2-D simulation with 50 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution.
"""

from nbody import rand, animate, save

# Random particles filename
filename = "random_particles_2D"

# Initializing a System() object via the rand() function
L = rand(50, x0 = (0,10), m = (1E9, 1E3))

T = 5E-2
dt = 1E-4

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
save(L, filename)

# Saving an animation of the system
animate(L, filename)
