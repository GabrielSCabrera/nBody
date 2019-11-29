"""
    A particle collides with a 10x10 lattice of particles with alternating
    charge.
"""

from nbody import lattice, Sphere, animate, save

# Lattice filename
filename = "collision_lattice_2D"

# Lattice parameters
lattice_kwargs = {"shape":(10,10), "mass":1E2, "charge":1E-6,
                  "distance":1, "radius":0.5}

# Initializing a System() object via the lattice() function
L = lattice(**lattice_kwargs)

x1, x2 = (-15, -5), (25, 25)
v1, v2 = (100, 90), (-110, -100)
w1, w2 = 0, 0
m1, m2 = 1E6, 1E6
q1, q2 = 1E-5, -1E-5
r1, r2 = 0.1, 0.1

# Creating two new Sphere() objects
P1 = Sphere(x1, v1, w1, m1, q1, r1)
P2 = Sphere(x2, v2, w2, m2, q2, r2)

# Adding the new particles to the System
L.add(P1)
L.add(P2)

T = 0.5
dt = 1E-3

# Solving for the given T and dt
L.solve(T, dt, collision = True)

# Saving the results to file
save(L, filename)

# Saving an animation of the system
animate(L, filename)
