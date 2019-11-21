import numpy as np
import sys

sys.path.append("..")
from particles import Simulation

# Filename for saving results
filename = "orbits"

N = 100

x = (0,80)
v = (0,2)
m = (1E10,1E3)
q = (0,0)
r = (1,0.1)

N = int(N)

p = 2

# Setting up the particle positions
x = np.random.normal(x[0], x[1], (N,p))

# Getting the position's unit vectors
x_norm = np.linalg.norm(x, axis = 0)
x_unit = x/x_norm

# Setting the velocities to be orthogonal to the position vectors
v_abs = np.random.normal(v[0], v[1], N)[:,None]
v_unit = x @ np.array([[0, -1],[1, 0]])
v = v_abs * v_unit

# Creating mass array, and accounting for negative values
m = np.random.normal(m[0], m[1], N)
m[m < 0] = np.abs(m[m < 0])
m[m == 0] = 1

# Creating charge array
q = np.random.normal(q[0], q[1], N)

# Creating radius array, and accounting for negative values
r = np.random.normal(r[0], r[1], N)
r[r < 0] = np.abs(r[r < 0])
r[r == 0] = 1

T = 1
dt = 1E-3

# Creating an instance of class Simulation
S = Simulation(x, v, m, q, r)

# Solving for the given T and dt
S.solve(T, dt, collision = True)

# Saving the results to file
S.save(filename)

# Saving an animation of the system
S.animate(filename)
