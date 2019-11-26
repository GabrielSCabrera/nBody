from nbody import Sphere, System, lattice, animate, save
import numpy as np

# Railgun filename
filename = "railgun"

sign = -1
N = 50
y_top = 4
y_bottom = 2
radii = 0.5
masses = 1E6
charges = 1E-3
particles = []
x0 = -100
for i in range(N):
    particles.append(Sphere((x0+2*i*radii+0.1, y_top), (0,0), masses, sign*charges, radii))
    particles.append(Sphere((x0+2*i*radii+0.1, y_bottom), (0,0), masses, sign*charges, radii))
    sign *= -1

P = Sphere([x0-3, (y_top + y_bottom)/2], [0.5,0], 1, 1E-3, 0.3)
S = lattice((5,5), 10, 0, 0.01, 1)

for i in particles:
    S.add(i)
S.add(P)

S.solve(1, 1E-3)
save(S, filename)
animate(S, filename)
