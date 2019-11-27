"""
    A 3-D simulation with 100 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution.
"""

from nbody import rand, animate, save

# Random particles filename
filename = "random_particles_3D"

# Initializing a System() object via the rand() function
L = rand(100, x0 = (0,10), p = 3, m = (1E9, 1E3))

T = 5E-2
dt = 1E-4

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
save(L, filename)

# Displaying an animation of the system
animate(L)
