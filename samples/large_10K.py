import sys

sys.path.append("..")
from particles import rand

# Filename for saving results
filename = "large_10K"

N = 1E4
# Initializing a Simulation() object via the rand() function
L = rand(N, x = (0,1E4), v = (0,5E4), m = (1E9,1E7), q = (0,2), r = (1E1,1E1))

T = 1.2
dt = 1E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
L.save(filename)

# Saving an animation of the system
L.animate(filename)
