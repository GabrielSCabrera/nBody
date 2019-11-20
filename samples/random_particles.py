import sys

sys.path.append("..")
from particles import rand

# Random particles filename
filename = "random_particles"

# Initializing a Simulation() object via the rand() function
L = rand(100)

T = 0.5
dt = 1E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
L.save(filename)

# Saving an animation of the system
L.animate(filename)
