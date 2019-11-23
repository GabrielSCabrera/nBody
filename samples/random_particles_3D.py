"""
    A 3-D simulation with 300 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution.
"""

import sys

sys.path.append("..")
from particles import rand

# Random particles filename
filename = "random_particles_3D"

# Initializing a Simulation() object via the rand() function
L = rand(300, x = (0,10) , p = 3, m = (1E9, 1E3))

T = 5E-2
dt = 1E-4

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
L.save(filename)

# Displaying an animation of the system
L.animate()
