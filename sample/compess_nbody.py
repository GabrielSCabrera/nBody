from nbody import Sphere, System, animate, save
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Creating two Sphere() objects
P1 = Sphere(x0 = (0.05, 0, 0), v0 = (0, 0.1, -0.1), m = 9.1E-31, q = -1.6E-19, r = 1)
P2 = Sphere(x0 = (0, 0, 0), v0 = (0, -0.1, 0.1), m = 1.67E-27, q = 1.6E-19,r = 1)

# Antall steg
N = 1E3

# Lager tidsarray
dt = 1E-6
T = N*dt

# Creating an instance of class System
S = System()

# Adding the new particle to the System
S.add(P1)
S.add(P2)

# Solving for the given T and dt
S.solve(T, dt, collision = False)

x0 = S.x[:,0,:]
x1 = S.x[:,1,:]

# Plotter resultat av bevegelser for partiklene
ax = plt.axes(projection='3d')
plt.title('Bevegelse til partiklene i tre dimensjoner')
ax.plot3D([x0[0,0]], [x0[0,1]], [x0[0,2]], marker = 'x', label='Elektron Pos 0')
ax.plot3D([x1[0,0]], [x1[0,1]], [x1[0,2]], marker = 'x', label='Proton Pos 0')
ax.plot3D(x0[:,0], x0[:,1], x0[:,2], color='#99CCFF', label='Elektron')
ax.plot3D(x1[:,0], x1[:,1], x1[:,2], color='#CC6666', label='Proton')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
