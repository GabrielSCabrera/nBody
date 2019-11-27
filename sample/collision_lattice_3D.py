"""
    A particle collides with a 5x5x5 lattice of particles with alternating
    charge.
"""

from nbody import lattice, Sphere, animate, save
import numpy as np

# Lattice filename
filename = "collision_lattice_3D"

# Lattice parameters
lattice_kwargs = {"shape":(5,5,5), "mass":1E3, "charge":1E-5,
                  "distance":0, "radius":0.5}

# Initializing a System() object via the lattice() function
L = lattice(**lattice_kwargs)

# Creating a new Sphere() object
x = (-10, -10, -10)
v = (40, 30, 34)
w = (0, 0, 0)
m, q, r = 50, 0, 0.25
P1 = Sphere(x, v, w, m, q, r)

# Adding the new particle to the System
L.add(P1)

T = 2
dt = 5E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
save(L, filename)

# Saving an animation of the system
animate(L)
