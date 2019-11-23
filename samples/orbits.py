"""
    A 2-D simulation with 300 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution; velocity is always
    tangential to each particle's position relative to the origin.

    In the origin, there is a mass much larger than the others.
"""

import numpy as np
import sys

sys.path.append("..")
from particles import Particle, Simulation

# Filename for saving results
filename = "orbits"

N = 500

x = (0,80)
v = (0.6,0.5)
m = (5,50)
q = (0,0)
r = (1,0.5)

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

# Creating a new Particle() object
x2 = (0, 0)
v2 = (0, 0)
m2, q2, r2 = 1E11, 0, 10
P1 = Particle(x2, v2, m2, q2, r2)

T = 50
dt = 0.1

# Creating an instance of class Simulation
S = Simulation(x, v, m, q, r)

# Adding the new particle to the Simulation
S.add(P1)

# Solving for the given T and dt
S.solve(T, dt, collision = True)

# Saving the results to file
S.save(filename)

# Saving an animation of the system
# S.animate()
S.animate(filename)
