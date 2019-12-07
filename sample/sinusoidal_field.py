"""
    A 3-D simulation with N**3 spheres in a container
"""

from nbody import lattice, animate, save, Field
import numpy as np

# Random particles filename
filename = "balls_in_a_box"

# Setting particle spacing
N = 4
d = 1
r = 0.5
d_max = N*(d+2*r)

# Calculating minimum needed boundaries
bounds = np.array([[0, d_max],[0, d_max],[0, d_max]])

# Scaling the boundaries to desired size (all factors must be >= 1)
f_x, f_y, f_z = 1,1,1
bounds *= np.array([f_x, f_y, f_z])[:,np.newaxis]
bounds[:,0] -= 1 + r
bounds[:,1] += 1 + r

# Initializing a System() object via the lattice() function
charge = 1E-8
L = lattice(shape = (N,N,N), mass = 0.1, charge = charge, distance = d,
            radius = r, flip = False)

# Creating a gravitational field
g = 0#9.81
def F_g(x, m):
    magnitude = g*m
    direction = np.zeros_like(x)
    direction[:,2] = -1
    return magnitude*direction

# Creating an electric field
frequency = d_max/7
charge_coeff = -5E-2/charge
c_t = 10
f_t = 5
def F_c(x, q, m, t):
    a = x.copy()
    a[:,0] = charge_coeff*q.flatten()*np.cos(frequency*(a[:,0]/2*np.pi))
    a[:,1] = charge_coeff*q.flatten()*np.sin(frequency*(a[:,1]/2*np.pi))
    a[:,2] = charge_coeff*q.flatten()*np.cos(frequency*(a[:,2]/2*np.pi))
    return a#*f_t*np.sin(t/c_t)**2

def F_tot(x, m, q, t):
    F_tot = F_g(x, m) + F_c(x, q, m, t)
    return F_tot

F = Field(F_c)
L.set_field(F)

T = 100
dt = 1E-1

# Solving for the given T and dt
L.solve(T, dt, collision = True, C_d = 0.75)#, bounds = bounds)

# Saving the results to file
save(L, filename)

# Displaying an animation of the system
animate(L)
