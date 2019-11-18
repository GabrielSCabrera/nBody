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

def random(N, x = (0,20), v = (0,10), m = (100,10), q = (0,1E-4), r = (1,1)):
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
            hh = int((eta//3600)%24)
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
        dd = int((dt//86400))
        hh = int(dt//3600)%24
        mm = int((dt//60)%60)
        ss = int(dt%60)
        msg = ""
        if dd > 0:
            msg += f"{dd:d} day(s) + "
        msg += f"{hh:02d}:{mm:02d}:{ss:02d}"
        print(f"\r\t100% – Total Time Elapsed {msg}")

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

    def test_GPU(self, collision):
        """
            If the GPU improves performance, returns True
            return False otherwise
        """
        if not cupy_imported:
            return False

        times = []

        for mod, GPU in zip([cp, np], [True, False]):
            a = mod.random.normal(0,0.5,(self.N, 3))
            b = mod.random.normal(0,0.5,(self.N, 1))

            t0 = time()
            foo = mod.linalg.norm(a*b, axis = 1)[:,np.newaxis]
            for n in range(0, 10):
                foo = self.arr_delete(a, n, GPU, axis = 0)
                foo = self.arr_delete(a, n, GPU, axis = 0)
                c = self.arr_delete(a, n, GPU, axis = 0) - a[n]
                d = mod.linalg.norm(c, axis = 1)[:,np.newaxis]
                e = d*a[n]/0.5
                f = mod.sum(e*c/d, axis = 0)
                if collision:
                    f = f + self._a_collision(b, b, a, a, c, d, n, 0.01, GPU,
                                              mod, 0.01)
                foo = a + 0.5*a
            times.append(time() - t0)
        return times[0] < times[1]

    def _a_collision(self, m, radius, x, v, r, r_norm, n, cf, GPU, mod, dt):
        m2 = self.arr_delete(m, n, GPU, axis = 0)
        v2 = self.arr_delete(v, n, GPU, axis = 0)
        radius_step = self.arr_delete(radius, n, GPU, axis = 0)
        f_col = v2*(m[n]-m2)/(m[n]+m2) + 2*m2*v[n]/(m[n]+m2)
        f_col = mod.linalg.norm(f_col, axis = 1)[:,np.newaxis]
        col_idx = (r_norm <= radius_step+radius[n]).flatten()
        step = f_col[col_idx]*r[col_idx]/r_norm[col_idx]
        return -cf*mod.sum(step, axis = 0)/dt

    def arr_delete(self, arr, n, GPU, axis):
        if n == 0:
            return arr[n+1:]
        elif n == arr.shape[0] - 1:
            return arr[:n]
        elif GPU is True:
            return cp.concatenate([arr[:n], arr[n+1:]])
        else:
            return np.delete(arr, n, axis = axis)

    def solve(self, T, dt, collision = True, GPU = None, debug = True):
        self.T, self.dt = T, dt

        # Selecting cupy or numpy depending on system/simulation parameters
        if GPU is None:
            GPU = self.test_GPU(collision)

        if cupy_imported is True and GPU:
            mod = cp
            GPU_active = "Active"
        else:
            mod = np
            GPU_active = "Inactive"

        if debug:
            msg = (f"\nSIMULATION INFO:\n\n\tParticles\t\t{self.N:d}\n\t"
                   f"Dimensions\t\t{self.p:d}\n\tT\t\t\t{T:g}\n\tdt\t\t\t"
                   f"{dt:g}\n\tSteps\t\t\t{T//dt:g}\n\tCUDA\t\t\t{GPU_active}"
                   f"\n\nSTATUS:\n")
            print(msg)

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
        col_cutoff = 0.5
        # Collision force coefficient
        cf = 0.5
        if debug:
            counter = Counter(2*steps*self.N)
        arange = np.arange(0,self.N,1)

        # Velocity Verlet Integration
        for m in range(1, steps):
            p_step = mod.linalg.norm(mass*v[m-1], axis = 1)[:,np.newaxis]
            for n in range(0, self.N):
                m1 = mass[n]
                m2 = self.arr_delete(mass, n, GPU, axis = 0)
                charge_step = self.arr_delete(charge, n, GPU, axis = 0)
                r = self.arr_delete(x[m-1], n, GPU, axis = 0) - x[m-1,n]
                r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                forces = G*m2 + k*charge_step*charge[n]/m1
                a = mod.sum(forces*r/r_norm, axis = 0)
                if collision:
                    a = a + self._a_collision(mass, radius, x[m-1], v[m-1], r,
                                              r_norm, n, cf, GPU, mod, dt)
                v_half[n] = v[m-1,n] + dt*0.5*a
                x[m,n] = x[m-1,n] + dt*v_half[n]
                if debug:
                    counter()

            p_step = mod.linalg.norm(mass*v[m], axis = 1)[:,np.newaxis]
            a = 0

            for n in range(0, self.N):
                m1 = mass[n]
                m2 = self.arr_delete(mass, n, GPU, axis = 0)
                charge_step = self.arr_delete(charge, n, GPU, axis = 0)
                r = self.arr_delete(x[m], n, GPU, axis = 0) - x[m,n]
                r_norm = mod.linalg.norm(r, axis = 1)[:,np.newaxis]
                forces = G*m2 + k*charge_step*charge[n]/m1
                a = mod.sum(forces*r/r_norm, axis = 0)
                if collision:
                    a = a + self._a_collision(mass, radius, x[m], v[m], r,
                                              r_norm, n, cf, GPU, mod, dt)
                v[m,n] = v_half[n] + dt*0.5*a
                if debug:
                    counter()

        del v_half
        if debug:
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
        plt.axis('off')
        fig.set_facecolor("k")

        colors_g = np.linalg.norm(self.v, axis = 2)
        colors_g = np.log(colors_g + np.min(colors_g) + 1E-5)
        cg_min = np.min(colors_g.flatten())
        cg_max = np.max(colors_g.flatten())
        colors_g = 1-((colors_g - cg_min)/(cg_max-cg_min))

        circles = []
        for i,j,k in zip(self.x[0], self.r, colors_g[0]):
            circles.append(Circle(tuple(i), j, color = (1,k,0)))
            ax.add_artist(circles[-1])

        def init():
            xlim = (np.min(self.x[0,:,0]-self.r), np.max(self.x[0,:,0]+self.r))
            ylim = (np.min(self.x[0,:,1]-self.r), np.max(self.x[0,:,1]+self.r))
            scale = xlim if xlim[1]-xlim[0] >= ylim[1]-ylim[0] else ylim
            scale = scale[1] - scale[0]
            if xlim[1]-xlim[0] >= ylim[1]-ylim[0]:
                y_mid = (ylim[1] - ylim[0])/2 + ylim[0]
                y_max = y_mid + scale/2
                y_min = y_mid - scale/2
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(y_min, y_max)
            else:
                x_mid = (xlim[1] - xlim[0])/2 + xlim[0]
                x_max = x_mid + scale/2
                x_min = x_mid - scale/2
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(ylim[0], ylim[1])
            return

        def update(m):
            for n,c in enumerate(circles):
                c.center = tuple(self.x[m,n])
                c.set_color((1,colors_g[m,n],0))
            xlim = (np.min(self.x[m,:,0]-self.r), np.max(self.x[m,:,0]+self.r))
            ylim = (np.min(self.x[m,:,1]-self.r), np.max(self.x[m,:,1]+self.r))
            scale = xlim if xlim[1]-xlim[0] >= ylim[1]-ylim[0] else ylim
            scale = scale[1] - scale[0]
            if xlim[1]-xlim[0] >= ylim[1]-ylim[0]:
                y_mid = (ylim[1] - ylim[0])/2 + ylim[0]
                y_max = y_mid + scale/2
                y_min = y_mid - scale/2
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(y_min, y_max)
            else:
                x_mid = (xlim[1] - xlim[0])/2 + xlim[0]
                x_max = x_mid + scale/2
                x_min = x_mid - scale/2
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(ylim[0], ylim[1])
            return

        frames = np.arange(0, self.x.shape[0], 1)
        ani = FuncAnimation(fig, update, frames = frames, init_func = init,
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
        print()

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

    filename = "test"
    dir = "saved"
    # L = load(f"{dir}/{filename}")
    # L = lattice((6, 6), 1E2, 1E-4, 2.01, 1)
    # P1 = Particle((-10, 6), (50, 0), 1E4, 0, 3)
    # P2 = Particle((-20, 6), (10, 0), 1E4, -1E-3, 1)
    # L.add(P1)
    # L.add(P2)
    L = random(100)
    L.solve(2, 1E-2, collision = True)
    L.save(f"{dir}/{filename}")
    # L.animate()
    L.animate(f"{dir}/{filename}/{filename}")
