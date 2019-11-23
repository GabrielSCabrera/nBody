"""
    A particle collides with a 5x5x5 lattice of particles with alternating
    charge.
"""

import numpy as np
import sys

sys.path.append("..")
from particles import lattice, Particle

# Lattice filename
filename = "collision_lattice_3D"

# Lattice parameters
lattice_kwargs = {"shape":(5,5,5), "mass":1E3, "absolute_charge":1E-5,
                  "distance":0, "radius":0.5}

# Initializing a Simulation() object via the lattice() function
L = lattice(**lattice_kwargs)

# Creating a new Particle() object
x = (-10, -10, -10)
v = (40, 30, 34)
m, q, r = 50, 0, 0.25
P1 = Particle(x, v, m, q, r)

# Adding the new particle to the Simulation
L.add(P1)

T = 2
dt = 5E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
L.save(filename)

# Saving an animation of the system
L.animate()
