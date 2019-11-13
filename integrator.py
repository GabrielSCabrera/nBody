from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os
try:
    import cupy as cp
    cupy_imported = True
except ImportError:
    cupy_imported = False
    warning_msg = ("\033[01mWARNING\033[m: Module \033[03mcupy\033[m is not "
                   "installed on this system. \033[03mcupy\033[m enables GPU "
                   "based acceleration through multiprocessing.")
    print(warning_msg)

np.random.seed(69420666)

def load(dirname):

    t = np.load(f"{dirname}/arr/t.npy")
    x = np.load(f"{dirname}/arr/x.npy")
    v = np.load(f"{dirname}/arr/v.npy")
    m = np.load(f"{dirname}/arr/m.npy")
    q = np.load(f"{dirname}/arr/q.npy")
    r = np.load(f"{dirname}/arr/r.npy")

    I = Integrator(x[0], v[0], m, q, r)
    I.t, I.x, I.v = t, x, v

    with open(f"{dirname}/metadata.dat") as infile:
        data = infile.read().split(" ")
        I.dt = data[0].split("=")[1]
        I.T = data[1].split("=")[1]

    return I

def lattice(shape, mass, absolute_charge, distance, radius):
    if len(shape) != 2:
        raise NotImplemented("Can currently only create 2-D lattice")

    ch_shape = list(shape)
    ch_shape[0] = np.max([shape[0]//2, 2])
    ch_shape[1] = np.max([shape[1]//2, 2])
    checker = [[1, 0] * ch_shape[0], [0, 1] * ch_shape[0]] * ch_shape[1]
    q = (absolute_charge*(np.kron(checker, np.ones((1, 1)))*2 - 1)).flatten()

    m = mass*np.ones_like(q)
    r = radius*np.ones_like(q)

    shape = list(shape)
    shape[0] = np.max([shape[0], 2])
    shape[1] = np.max([shape[1], 2])
    N = np.prod(shape)

    v = np.zeros((N,2))

    x_row = np.linspace(0, (shape[0]-1)*distance, shape[0])
    y_row = np.linspace(0, (shape[1]-1)*distance, shape[1])
    X,Y = np.meshgrid(x_row, y_row)
    x = np.zeros_like(v)
    x[:,0], x[:,1] = X.flatten(), Y.flatten()

    return Integrator(x, v, m, q, r)

def random(N, x = (0,100), v = (0,10), m = (1E3,5E2), q = (0,1E-5), r = (0,2)):
    N = int(N)
    p = 2
    x = np.random.normal(x[0], x[1], (N,p))
    v = np.random.normal(v[0], v[1], (N,p))
    m = np.random.normal(m[0], m[1], N)
    m[m < 0] = np.abs(m[m < 0])
    m[m == 0] = 1
    q = np.random.normal(q[0], q[1], N)
    r = np.random.normal(r[0], r[1], N)
    return Integrator(x, v, m, q, r)

class Counter:

    def __init__(self, tot_iter):
        self.counter = 0
        self.t0 = time()
        self.perc = 0
        self.tot_iter = tot_iter
        self.times = np.zeros(tot_iter)
        self.dt = 0
        print(f"\t{0:>3d}%", end = "")

    def __call__(self):
        self.counter += 1
        new_perc = int(100*self.counter/self.tot_iter)
        self.times[self.counter-1] = time()
        if int(time() - self.t0) > self.dt and self.counter > 1:
            self.perc = new_perc
            t_avg = np.mean(np.diff(self.times[:self.counter]))
            eta = t_avg*(self.tot_iter - self.counter)
            dd = int((eta//86400))
            hh = int((eta//3600)%60)
            mm = int((eta//60)%60)
            ss = int(eta%60)
            msg = f"\r\t{self.perc:>3d}% – "
            if dd > 0:
                msg += f"{dd:d} day(s) + "
            msg += f"{hh:02d}:{mm:02d}:{ss:02d}"
            print(msg, end = "")
        self.dt = time() - self.t0

    def close(self):
        dt = time() - self.t0
        hh = int(dt//3600)
        mm = int((dt//60)%60)
        ss = int(dt%60)
        print(f"\r\t100% – Total Time Elapsed {hh:02d}:{mm:02d}:{ss:02d}")

class Particle:

    def __init__(self, position, velocity, mass, charge, radius):
        """
            –– INPUT ARGUMENTS ––––––––––––––––
                    position    –   numerical array of shape (p,)
                    velocity    –   numerical array of shape (p,)
                    mass        –   number
                    charge      –   number

            –– NOT YET IMPLEMENTED ––––––––––––
                    radius      –   numerical array of shape (N,)

            –– FOR S.I. UNITS –––––––––––––––––
                    position    –   meters
                    velocity    –   meters/second
                    mass        –   kilograms
                    charge      –   coulombs
                    radius      –   meters
        """
        # Initializing Position, Velocity, Mass, and Radius
        self.x = np.array(position)
        self.v = np.array(velocity)
        self.m = np.array(mass)
        self.q = np.array(charge)
        self.r = np.array(radius)

class Integrator:

    def __init__(self, positions, velocities, masses, charges, radii):
        """
            –– INPUT ARGUMENTS ––––––––––––––––
                    positions   –   numerical array of shape (N,p)
                    velocities  –   numerical array of shape (N,p)
                    masses      –   numerical array of shape (N,)
                    charges     –   numerical array of shape (N,)

            –– NOT YET IMPLEMENTED ––––––––––––
                    radii       –   numerical array of shape (N,)

            –– FOR S.I. UNITS –––––––––––––––––
                    position    –   meters
                    velocity    –   meters/second
                    mass        –   kilograms
                    charge      –   coulombs
                    radius      –   meters
        """
        # Checking that positions and velocities are of the same shape, and
        # that masses and radii are equal on axis zero to positions and
        # velocities. Also checks if positions and velocities are 2-D, and if
        # masses and radii are 1-D.

        condition1 = not np.array_equal(positions.shape,  velocities.shape)
        condition2 = positions.shape[0] != masses.shape[0]
        condition3 = positions.shape[0] != charges.shape[0]
        condition4 = positions.ndim != 2 or masses.ndim != 1 or charges.ndim != 1

        if condition1 or condition2 or condition3 or condition4:
            msg = ("Invalid arrays passed to Integration object.\nArguments "
            "<positions> and <velocities> must be 2-D and of shape (N,3), "
            "while arguments <masses>, <charges>, and <radii> must be 1-D and "
            "of length N.")
            raise ValueError(msg)

        # Initializing Positions, Velocities, Masses, and Radii
        self.x0 = positions
        self.v0 = velocities
        self.m = masses
        self.q = charges
        self.r = radii

        self.N = self.x0.shape[0]
        self.p = self.x0.shape[1]

        self.attribute_reset()

    def attribute_reset(self):
        # Cleans out all attributes in case of reset
        if not hasattr(self, '_dir_backup'):
            self._dir_backup = self.__dir__().copy
        else:
            for var in self.__dir__():
                if var not in self._dir_backup() and var != "_dir_backup":
                    delattr(self, var)

    def add(self, particle):
        self.N += 1
        self.x0 = np.vstack([self.x0, particle.x])
        self.v0 = np.vstack([self.v0, particle.v])
        self.m = np.concatenate([self.m, [particle.m]])
        self.q = np.concatenate([self.q, [particle.q]])
        self.r = np.concatenate([self.r, [particle.r]])
        self.attribute_reset()

    def test_GPU(self):
        """
            If the GPU improves performance, returns True
            return False otherwise
        """
        if cupy_imported:
            a = np.random.normal(0,0.5,(self.N, 3))
            b = np.random.normal(0,0.5,(self.N, 3))

            a_gpu = cp.array(a.copy())
            b_gpu = cp.array(b.copy())

            t0 = time()
            for i in range(100):
                c = a * b - a[0]
                c += c*0.1/2**3
                np.sum(c, axis = 0)
                np.linalg.norm(c)
            t1 = time()

            t0_gpu = time()
            for i in range(100):
                c_gpu = a_gpu * b_gpu - a_gpu[0]
                c_gpu += c_gpu*0.1/2**3
                cp.sum(c, axis = 0)
                cp.linalg.norm(c_gpu)
            t1_gpu = time()

            return (t1_gpu - t0_gpu) < (t1 - t0)
        return False

    def solve(self, T, dt, collision = True, GPU = None):
        self.T, self.dt = T, dt

        # Selecting cupy or numpy depending on system/simulation parameters
        if GPU is None:
            GPU = self.test_GPU()

        if cupy_imported is True and GPU:
            mod = cp
        else:
            mod = np

        steps = int(T//dt)
        if steps*dt < T:
            steps += 1
        T = steps*dt

        x = mod.zeros((steps, self.N, self.p))
        v = mod.zeros((steps, self.N, self.p))
        mass = mod.array(self.m[:,mod.newaxis])
        charge = mod.array(self.q[:,mod.newaxis])
        radius = mod.array(self.r[:,mod.newaxis])

        # Inserting initial conditions
        x[0] = mod.array(self.x0)
        v[0] = mod.array(self.v0)

        # Allocating Memory to Temporary Variables
        v_half = mod.zeros((self.N,self.p))

        # Scientific Constants
        G = 6.67408E-11
        k = 8.9875517887E9
        eps = 2# Depth of collision potential well
        R1,R2 = mod.meshgrid(radius, radius)
        A = (4*eps*(R1+R2)**12)[:,:,mod.newaxis]
        B = (4*eps*(R1+R2)**6)[:,:,mod.newaxis]
        col_cutoff = 0.99
        cf = 0.5# Collision force coefficient
        counter = Counter(2*steps*self.N)

        # Velocity Verlet Integration
        for m in range(1, steps):
            p_step = mod.linalg.norm(mass*v[m-1], axis = 1)[:,np.newaxis]
            for n in range(0, self.N):
                a = 0
                if n > 0:
                    r = x[m-1,:n]-x[m-1,n]
                    r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                    forces = G*mass[:n] + k*charge[:n]*charge[n]/mass[n]
                    a = a + mod.sum(forces*r/r_norm, axis = 0)
                    if collision:
                        m1 = mass[n]
                        m2 = mass[:n]
                        f_col = v[m-1,n]*(1 - 2*m2/(m1+m2))
                        f_col = f_col + v[m-1,:n]*(m2/m1)*(1 - (m2-m1)/(m1+m2))
                        f_col = mod.linalg.norm(f_col, axis = 1)[:,np.newaxis]
                        col_idx = (r_norm <= radius[:n]+radius[n]).flatten()
                        step = f_col*r/r_norm
                        a = a - cf*mod.sum(step[col_idx], axis = 0)/dt

                if n < self.N - 1:
                    r = x[m-1,n+1:]-x[m-1,n]
                    r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                    forces = G*mass[n+1:] + k*charge[n+1:]*charge[n]/mass[n]
                    a = a + mod.sum(forces*r/r_norm, axis = 0)
                    if collision:
                        m1 = mass[n]
                        m2 = mass[n+1:]
                        f_col = v[m-1,n]*(1 - 2*m2/(m1+m2))
                        f_col = f_col + v[m-1,n+1:]*(m2/m1)*(1 - (m2-m1)/(m1+m2))
                        f_col = mod.linalg.norm(f_col, axis = 1)[:,np.newaxis]
                        col_idx = (r_norm <= radius[n+1:]+radius[n]).flatten()
                        step = f_col*r/r_norm
                        a = a - cf*mod.sum(step[col_idx], axis = 0)/dt

                v_half[n] = v[m-1,n] + dt*0.5*a
                x[m,n] = x[m-1,n] + dt*v_half[n]
                counter()

            p_step = mod.linalg.norm(mass*v[m], axis = 1)[:,np.newaxis]
            for n in range(0, self.N):
                a = 0
                if n > 0:
                    r = x[m,:n]-x[m,n]
                    r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                    forces = G*mass[:n] + k*charge[:n]*charge[n]/mass[n]
                    a = a + mod.sum(forces*r/r_norm, axis = 0)
                    if collision:
                        m1 = mass[n]
                        m2 = mass[:n]
                        f_col = v[m,n]*(1 - 2*m2/(m1+m2))
                        f_col = f_col + v[m,:n]*(m2/m1)*(1 - (m2-m1)/(m1+m2))
                        f_col = mod.linalg.norm(f_col, axis = 1)[:,np.newaxis]
                        col_idx = (r_norm <= radius[:n]+radius[n]).flatten()
                        step = f_col*r/r_norm
                        a = a - cf*mod.sum(step[col_idx], axis = 0)/dt

                if n < self.N - 1:
                    r = x[m,n+1:]-x[m,n]
                    r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                    forces = G*mass[n+1:] + k*charge[n+1:]*charge[n]/mass[n]
                    a = a + mod.sum(forces*r/r_norm, axis = 0)
                    if collision:
                        m1 = mass[n]
                        m2 = mass[n+1:]
                        f_col = v[m,n]*(1 - 2*m2/(m1+m2))
                        f_col = f_col + v[m,n+1:]*(m2/m1)*(1 - (m2-m1)/(m1+m2))
                        f_col = mod.linalg.norm(f_col, axis = 1)[:,np.newaxis]
                        col_idx = (r_norm <= radius[n+1:]+radius[n]).flatten()
                        step = f_col*r/r_norm
                        a = a - cf*mod.sum(step[col_idx], axis = 0)/dt
                v[m,n] = v_half[n] + dt*0.5*a
                counter()

        del v_half
        counter.close()

        # Converting back to numpy, if cupy is used
        if cupy_imported is True and GPU:
            x = cp.asnumpy(x)
            v = cp.asnumpy(v)

        self.t = np.linspace(0, T, steps)
        self.x = x
        self.v = v

    def animate(self, savename = None):

        if self.p != 2:
            raise NotImplemented("Can currently only perform 2-D animation")

        if np.max(self.m) == np.min(self.m):
            sizes = 20*np.ones_like(self.m)
        else:
            denom = np.max(self.m) - np.min(self.m)
            sizes = 30*(self.m - np.min(self.m))/denom + 1

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        fig.set_size_inches(15, 9)
        # ln = plt.scatter(self.x[0,:,0], self.x[0,:,1], color=np.random.rand(self.N, 3))
        # ln.set_sizes(sizes)
        plt.axis('off')
        fig.set_facecolor("k")

        circles = []
        for i,j,k,l,m in zip(self.x[0], self.v[0], self.m, self.q, self.r):
            circles.append(Circle(tuple(i), m, color = np.random.rand(3)))#, color = "w"))
            # circles.append(Circle(tuple(i), m, color = "w"))
            ax.add_artist(circles[-1])

        def init():
            xlim = (np.min(self.x[:,:,0]), np.max(self.x[:,:,0]))
            ylim = (np.min(self.x[:,:,1]), np.max(self.x[:,:,1]))
            limit = xlim if xlim[1]-xlim[0] >= ylim[1]-ylim[0] else ylim
            ax.set_xlim(limit[0], limit[1])
            ax.set_ylim(limit[0], limit[1])
            return

        def update(frame):
            for n,c in enumerate(circles):
                c.center = tuple(frame[n])
            return

        # def update(frame):
        #     ln.set_offsets(frame)
        #     return ln

        ani = FuncAnimation(fig, update, frames = self.x, init_func = init,
                            blit = False, interval = 25)
        if savename is None:
            plt.show()
        else:
            Writer = writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='gabrielscabrera'),
                            bitrate=2500)
            ani.save(f"{savename}.mp4", writer=writer,
                     savefig_kwargs={'facecolor':fig.get_facecolor(),
                                     'repeat':True})
            plt.close()

    def save(self, dirname = "nBody_save_"):
        if dirname[-1] == "_":
            ID = 0.0
            while True:
                ID_text = f"{ID:03.0f}"
                if os.path.isdir(dirname + ID_text):
                    ID += 1
                else:
                    dirname = dirname + ID_text
                    os.mkdir(dirname)
                    os.mkdir(dirname + "/arr")
                    break
        elif not os.path.isdir(dirname):
            os.mkdir(dirname)
            os.mkdir(dirname + "/arr")
        else:
            if not os.path.isdir(dirname + "/arr"):
                os.mkdir(dirname + "/arr")

        np.save(f"{dirname}/arr/t", self.t)
        np.save(f"{dirname}/arr/x", self.x)
        np.save(f"{dirname}/arr/v", self.v)
        np.save(f"{dirname}/arr/m", self.m)
        np.save(f"{dirname}/arr/q", self.q)
        np.save(f"{dirname}/arr/r", self.r)

        with open(f"{dirname}/metadata.dat", "w+") as outfile:
            outfile.write(f"dt={self.dt} T={self.T}")

if __name__ == "__main__":

    # I = load("large_15K")
    # I.animate("test2")
    # L = lattice((6, 6), 1E3, 1E-4, 2.5, 1)
    # P1 = Particle((-5, 6), (10, 0), 1E3, 0, 2)
    # P2 = Particle((-20, 6), (10, 0), 1E4, -1E-3, 1)
    # L.add(P1)
    # L.add(P2)
    L = random(1E3)
    L.solve(10, 1E-2)
    L.save("small_1K")
    L.animate("small_1K/small_1K")
