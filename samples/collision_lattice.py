import sys

sys.path.append("..")
from particles import lattice, Particle

# Lattice filename
filename = "collision_lattice"

# Lattice parameters
lattice_kwargs = {"shape":(10,10), "mass":1E2, "absolute_charge":1E-5,
                  "distance":1, "radius":0.5}

# Initializing a Simulation() object via the lattice() function
L = lattice(**lattice_kwargs)

x1, x2 = (-20, -20), (30, 30)
v1, v2 = (100, 100), (-100, -100)
m1, m2 = 1E6, 1E6
q1, q2 = 1E-5, -1E-5
r1, r2 = 0.1, 0.1

# Creating two new Particle() objects
P1 = Particle(x1, v1, m1, q1, r1)
P2 = Particle(x2, v2, m2, q2, r2)

# Adding the new particles to the Simulation
L.add(P1)
L.add(P2)

T = 1
dt = 1E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
L.save(filename)

# Saving an animation of the system
L.animate(filename)
