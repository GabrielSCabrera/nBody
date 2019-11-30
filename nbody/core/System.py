"""
    Models the trajectories of spheres that interact with one another
    through gravitational forces, Coulomb interactions, and collisions.
"""

from ..utils.checking import check_numerical_return_array
from ..utils.checking import check_ndim_return_array
from ..utils.exceptions import DimensionError
from ..utils.validation import validate_time
from ..utils.exceptions import PhysicsError
from ..utils.exceptions import ShapeError
from ..utils.Counter import Counter
from .Sphere import Sphere

from time import time
import numpy as np
import os

try:
    import cupy as cp
    cupy_imported = True
except ImportError:
    cupy_imported = False

class System:

    def __init__(self):
        self.N = None
        self.p = None
        self.x0 = None
        self.v0 = None
        self.w0 = None
        self.m = None
        self.q = None
        self.r = None
        self.attribute_reset()

    def attribute_reset(self):
        # Saves instance state on run 1, resets to this state on runs 2+
        if not hasattr(self, '_dir_backup'):
            self._dir_backup = self.__dir__().copy
        else:
            for var in self.__dir__():
                if var not in self._dir_backup() and var != "_dir_backup":
                    delattr(self, var)

    def add(self, spheres):
        """
            Adds one or more new Spheres to the system.

            Argument 'spheres' should be a 'Sphere' or sequence thereof.
        """

        # Type Checking
        if isinstance(spheres, Sphere):
            spheres = [spheres]
        elif isinstance(spheres, (list, tuple, np.ndarray)):
            for n,sphere in enumerate(spheres):
                if not isinstance(sphere, Sphere):
                    msg = (f"Element {n:d} of 'spheres' in 'System.add' "
                           f"is of type {str(type(sphere))}; all elements "
                           f"in sequence must be 'Sphere' objects.")
                    raise TypeError(msg)
        else:
            msg = (f"Argument 'spheres' in 'System.add' is of type "
                   f"{str(type(spheres))}; expected a 'Sphere' object, or "
                   f"list containing compatible 'Sphere' objects.")
            raise TypeError(msg)

        # Dimension Checking
        for n,sphere in enumerate(spheres):
            if n == 0:
                p = sphere.p
            else:
                if sphere.p != p:
                    raise DimensionError(p, sphere.p)

        for sphere in spheres:
            # If the System is empty, initializes it
            if self.N is None:
                self.x0 = np.array(sphere.x0)
                self.v0 = np.array(sphere.v0)
                self.w0 = np.array(sphere.w0)
                self.m = np.array(sphere.m)
                self.q = np.array(sphere.q)
                self.r = np.array(sphere.r)
                self.N = 1
                self.p = self.x0.shape[1]
            else:
                # Updating the number of spheres
                self.N += 1

                # Updating the initial conditions for position, velocity, and
                # angular velocity
                self.x0 = np.vstack([self.x0, sphere.x0])
                self.v0 = np.vstack([self.v0, sphere.v0])
                self.w0 = np.vstack([self.w0, sphere.w0])
                # Including the new sphere's mass, charge, and radius
                self.m = np.vstack([self.m, sphere.m])
                self.q = np.vstack([self.q, sphere.q])
                self.r = np.vstack([self.r, sphere.r])

        # Resetting the object to its original state, including the new data
        self.attribute_reset()

    def _test_GPU(self, collision):
        """
            Runs several iterations of an algorithm similar to the one used in
            the solve() method to gauge whether or not the GPU should be used

            If the CPU performs best or cupy isn't installed, returns False
            If the GPU performs best, returns True
        """
        if not cupy_imported:
            return False

        times = []

        # An algorithm similar to that used in the solve() method
        for mod, GPU in zip([cp, np], [True, False]):
            a = mod.random.normal(0, 0.5, (self.N, 1))
            b = mod.random.normal(0, 0.5, (self.N, self.p))

            t0 = time()
            foo = mod.linalg.norm(a*b, axis = 1)[:,np.newaxis]
            for n in range(0, 10):
                c = self._arr_del(arr = a, n = 0, GPU = GPU, axis = 0)
                c = self._arr_del(arr = a, n = 0, GPU = GPU, axis = 0)
                c = self._arr_del(arr = a, n = 0, GPU = GPU, axis = 0)
                d = self._arr_del(arr = b, n = 0, GPU = GPU, axis = 0)
                c = mod.linalg.norm(d, axis = 1)[:,np.newaxis]
                e = self._a_inv_square(m1 = 1.1, m2 = c, d2 = d, dn = c,
                                       q1 = 1.1, q2 = 1.1, G = 1.1, k = 1.1,
                                       mod = mod)
                if collision:
                    a = a + self._a_collision(m1 = 1.1, m2 = c, r1 = 1.1,
                                              r2 = c, v1 = d[0], v2 = d,
                                              w1 = d[0], w2 = d, d2 = d,
                                              dn = c, cf = 1.1, mod = mod,
                                              dt = 1.1)
                c = 0.5*(c+1)
            times.append(time() - t0)
        return times[0] < times[1]

    def _a_inv_square(self, m1, m2, d2, dn, q1, q2, G, k, mod):
        """
            Calculates the total acceleration on a sphere due to
            gravitational and Coulomb interactions, from all other spheres

            m1  – mass of current sphere
            m2  – masses of all other spheres
            q1  – charge of current sphere
            q2  – charges of all other spheres
            d2  – vectors pointing from all spheres, toward the current one
            dn  – distances between all spheres and current sphere

            G   – universal gravitational constant: 6.67430E−11
            k   – electrostatic constant: 8.9875517887E9
        """
        # Calculating gravitational acceleration
        a_g = G*m2
        # Calculating Coulomb acceleration
        a_c = k*q2*q1/m1
        return mod.sum((a_g + a_c)*d2/dn, axis = 0)

    def _a_collision(self, m1, m2, r1, r2, v1, v2, w1, w2, d2, dn, cf, mod, dt):
        """
            Calculates the total acceleration of a sphere due to collisions
            with all the other spheres

            m1  – mass of current sphere
            m2  – masses of all other spheres
            r1  – radius of current sphere
            r2  – radii of all other spheres
            v1  – velocity of current sphere
            v2  – velocities of all other spheres
            d2  – vectors pointing from all spheres, toward the current one
            dn  – distances between all spheres and current sphere

            cf  – force coefficient for collision
            mod – cupy if the GPU is active, numpy otherwise
            dt  – integration time-step
        """
        # Indices of spheres that are colliding with current sphere
        idx1 = (dn <= r2+r1).flatten()
        # Indices of spheres who are moving toward current sphere
        idx2 = np.less(np.sum(d2*(v2-v1), axis = 1),0)
        # Combining boolean arrays
        idx = np.logical_and(idx1, idx2)
        # Find acceleration by conservation laws for elastic collisions
        dv = 2*m2[idx]/(m1+m2[idx])*d2[idx]/dn[idx]**2
        dv *= np.sum(d2[idx]*(v2[idx]-v1), axis = 1)[:,np.newaxis]
        return np.sum(dv, axis = 0)/dt

    def _bounds(self, x, v, r, mod):
        """
            Bounding box that surrounds the system at limits denoted in
            attribute 'self.bounds'
        """
        low = mod.less_equal(x - self.bounds[:,0] - r, 0)
        high = mod.less_equal(self.bounds[:,1] - x + r, 0)
        low = mod.logical_and(low, mod.less(v, 0))
        high = mod.logical_and(high, mod.greater(v, 0))
        flip = mod.logical_or(low, high)
        v[flip] = -v[flip]
        return v

    def _arr_del(self, arr, n, GPU, axis):
        """
            Deletes an element from an array, for both cupy and numpy
        """
        if n == 0:
            return arr[n+1:]
        elif n == arr.shape[0] - 1:
            return arr[:n]
        elif GPU is True:
            return cp.concatenate([arr[:n], arr[n+1:]])
        else:
            return np.delete(arr, n, axis = axis)

    def simulation_info(self):
        """
            Returns a string of information about the ongoing simulation
        """
        if self.GPU_active is True:
            GPU = "Active"
        else:
            GPU = "Inactive"

        if self.collision is True:
            col = "Active"
        else:
            col = "Inactive"

        msg = (f"\nSIMULATION INFO:\n\n\tParticles\t\t{self.N:d}\n\t"
                   f"Dimensions\t\t{self.p:d}\n\tT\t\t\t{self.T:g}\n\tdt\t\t\t"
                   f"{self.dt:g}\n\tSteps\t\t\t{self.T//self.dt:g}\n\tCUDA"
                   f"\t\t\t{GPU}\n\tCollisions\t\t{col}")
        return msg

    def solve(self, T, dt = None, GPU = None, debug = True, bounds = None, collision = True):
        # Auto-selecting dt if None
        if dt is None:
            dt = T/500
        else:
            dt = validate_time(dt)
        T = validate_time(T)

        # Confirming that 'bounds' is a numerical array of shape (p,2)
        if bounds is not None:
            bounds = check_numerical_return_array(bounds)
            condition1 = not np.array_equal(bounds.shape, (self.p, 2))
            condition2 = np.all(np.greater(bounds[:,0], bounds[:,1]))
            if condition1 or condition2:
                msg = (f"Argument 'bounds' must be None or shape (p,2) with "
                       f"the first axis representing a dimension, and the "
                       f"second representing the lower and upper bounds of "
                       f"the box, respectively, in that dimension.")
                raise ShapeError(msg)

            # Checking that the boundaries contain the entire system
            low = self.x0 - bounds[:,0] - self.r
            high = bounds[:,1] - self.x0 + self.r
            out = np.logical_or(np.less_equal(low, 0), np.less_equal(high, 0))
            if np.any(out == True):
                msg = (f"Must extend bounding boxes so that the entire system "
                       f"is contained within its boundaries.")
                raise PhysicsError(msg)
        self.bounds = bounds

        # Auto-selecting cupy or numpy depending on system/simulation
        if GPU is None:
            GPU = self._test_GPU(collision)

        # If GPU is selected or overwritten, uses cupy.  Uses numpy otherwise
        if cupy_imported is True and GPU:
            mod = cp
            self.GPU_active = True
        else:
            mod = np
            self.GPU_active = False

        # Calculating number of steps to take in integration
        steps = int(T//dt)
        if steps*dt < T:
            steps += 1
        T = steps*dt

        self.T, self.dt = T, dt
        self.collision = collision

        # Prints information on the simulation taking place
        if debug:
            print(self.simulation_info(), end = "\n\n")

        # Initializing empty arrays for positions and velocities
        x = mod.zeros((steps, self.N, self.p))
        v = mod.zeros((steps, self.N, self.p))
        w = mod.zeros((steps, self.N, self.p))

        # Loading masses, charges, and radii from attributes
        mass = mod.array(self.m)
        charge = mod.array(self.q)
        radius = mod.array(self.r)

        # Inserting initial conditions
        x[0] = mod.array(self.x0)
        v[0] = mod.array(self.v0)
        w[0] = mod.array(self.w0)

        # Allocating memory for temporary variables
        v_half = mod.zeros((self.N, self.p))
        w_half = mod.zeros((self.N, self.p))

        # Universal gravitational constant
        G = 6.67430E-11
        # Electrostatic constant
        k = 8.9875517887E9
        # Collision force coefficient
        cf = 1

        # Initialize countdown timer
        if debug:
            counter = Counter(2*steps*self.N)

        # Velocity Verlet Integration
        for m in range(1, steps):
            # Loop over each sphere
            for n in range(0, self.N):
                # Masses of all spheres except the current one
                m2 = self._arr_del(arr = mass, n = n, GPU = GPU, axis = 0)
                # Charges of all spheres except the current one
                q2 = self._arr_del(arr = charge, n = n, GPU = GPU, axis = 0)
                # Velocities of all spheres except the current one
                v2 = self._arr_del(arr = v[m-1], n = n, GPU = GPU, axis = 0)
                # Angular velocities of all spheres except the current one
                w2 = self._arr_del(arr = w[m-1], n = n, GPU = GPU, axis = 0)
                # Radii of all spheres except the current one
                r2 = self._arr_del(arr = radius, n = n, GPU = GPU, axis = 0)
                # Vectors pointing from each sphere, toward the current one
                d2 = self._arr_del(arr = x[m-1], n = n, GPU = GPU, axis = 0)\
                                   - x[m-1,n]
                # Distances between current sphere, and all others
                dn = mod.linalg.norm(d2, axis = 1)[:,np.newaxis]
                # Sum over total gravitational and Coulomb accelerations
                a = self._a_inv_square(m1 = mass[n], m2 = m2, d2 = d2, dn = dn,
                                       q1 = charge[n], q2 = q2, G = G, k = k,
                                       mod = mod)
                if self.collision:
                    # Including acceleration from intersphere collisions
                    a = a + self._a_collision(m1 = mass[n], m2 = m2,
                        r1 = radius[n], r2 = r2, v1 = v[m-1,n], v2 = v2,
                        w1 = w[m-1,n], w2 = w2, d2 = d2, dn = dn, cf = cf,
                        mod = mod, dt = dt)

                # Verlet half-step velocity
                v_half[n] = v[m-1,n] + dt*0.5*a
                # w_half[n] = w[m-1,n] + dt*0.5*a

                # Display countdown timer
                if debug:
                    counter()

            # Updating new position
            x[m] = x[m-1] + dt*v_half

            # Loop over each sphere
            for n in range(0, self.N):
                # Masses of all spheres except the current one
                m2 = self._arr_del(arr = mass, n = n, GPU = GPU, axis = 0)
                # Charges of all spheres except the current one
                q2 = self._arr_del(arr = charge, n = n, GPU = GPU, axis = 0)
                # Velocities of all spheres except the current one
                v2 = self._arr_del(arr = v_half, n = n, GPU = GPU, axis = 0)
                # Angular velocities of all spheres except the current one
                w2 = self._arr_del(arr = w[m], n = n, GPU = GPU, axis = 0)
                # Radii of all spheres except the current one
                r2 = self._arr_del(arr = radius, n = n, GPU = GPU, axis = 0)
                # Vectors pointing from each sphere, toward the current one
                d2 = self._arr_del(arr = x[m], n = n, GPU = GPU, axis = 0)\
                                   - x[m,n]
                # Distances between current sphere, and all others
                dn = mod.linalg.norm(d2, axis = 1)[:,np.newaxis]
                # Sum over total gravitational and Coulomb accelerations
                a = self._a_inv_square(m1 = mass[n], m2 = m2, d2 = d2, dn = dn,
                                       q1 = charge[n], q2 = q2, G = G, k = k,
                                       mod = mod)
                if collision:
                    # Including acceleration from intersphere collisions
                    a = a + self._a_collision(m1 = mass[n], m2 = m2,
                        r1 = radius[n], r2 = r2, v1 = v_half[n], v2 = v2,
                        w1 = w[m,n], w2 = w2, d2 = d2, dn = dn, cf = cf,
                        mod = mod, dt = dt)

                # Updating new velocity
                v[m,n] = v_half[n] + dt*0.5*a

                # Display countdown timer
                if debug:
                    counter()

            # Reversing velocities at boundaries
            if self.bounds is not None:
                v[m] = self._bounds(x[m], v[m], radius, mod)

        # Display total time elapsed
        if debug:
            counter.close()

        # Converts back to numpy, if cupy is used
        if cupy_imported is True and GPU:
            x = cp.asnumpy(x)
            v = cp.asnumpy(v)

        # Saving array of time-steps
        self.t = np.linspace(0, T, steps)
        self.x = x
        self.v = v
        self.w = w

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
