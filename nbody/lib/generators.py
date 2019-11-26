from ..core import System

import numpy as np

def spheres(x0, v0, m, q, r):
    """

        Returns a System instance containing the initial conditions passed to
        this function.

        All arguments must be numerical arrays, and all values must be in
        S.I. units.

        –– INPUT ARGUMENTS ––––––––––––––––––––––––––––––––––––––––––––––––

                ARG     DESCRIPTION     SHAPE       S.I. UNITS

                x0  –   positions   –   (N,p)   –   meters
                v0  –   velocities  –   (N,p)   –   meters/second
                m   –   masses      –   (N,)    –   kilograms
                q   –   charges     –   (N,)    –   coulombs
                r   –   radii       –   (N,)    –   meters

    """
    S = System()
    S.x0 = x0
    S.v0 = v0
    S.m = m
    S.q = q
    S.r = r
    S.N = x0.shape[0]
    S.p = x0.shape[1]
    return S

def lattice(shape, mass, charge, distance, radius):
    shape = np.array(shape)
    p = shape.ndim
    N = np.prod(shape)

    # Setting lattice positions
    ctr_dist = 2*radius + distance
    arrays = []
    for i in shape:
        if i == 1:
            extension = np.array([0])
        else:
            extension = np.arange(0, ctr_dist*i, ctr_dist)
        arrays.append(extension)

    x0 = np.meshgrid(*arrays)
    for n,i in enumerate(x0):
        x0[n] = np.reshape(i, N)

    x0 = np.array(x0).T

    # Setting lattice velocities (zero)
    v0 = np.zeros_like(x0)
    # Setting lattice masses
    m = np.ones(N)*mass

    # Setting lattice charges
    bool_shape = ((shape + 1) % 2).astype(bool)
    slices = [slice(i) for i in shape] if shape.ndim > 0 else slice(shape)
    shape[bool_shape] += 1
    q = np.ones(np.prod(shape))
    q[::2] = -1
    q = q.reshape(shape)
    q = q[tuple(slices)]*charge
    q = q.reshape(N)

    # Setting lattice radii
    r = np.ones(N)*radius

    return spheres(x0, v0, m, q, r)

def rand(N, p = 2, x0 = (0,100), v0 = (0,100), m = (1E7,1E5), q = (0,1E-5), r = (1,0.1)):
    N = int(N)
    x0 = np.random.normal(x0[0], x0[1], (N,p))
    v0 = np.random.normal(v0[0], v0[1], (N,p))
    m = np.random.normal(m[0], m[1], N)
    m[m < 0] = np.abs(m[m < 0])
    m[m == 0] = 1
    q = np.random.normal(q[0], q[1], N)
    r = np.random.normal(r[0], r[1], N)
    r[r < 0] = np.abs(r[r < 0])
    r[r == 0] = 1
    return spheres(x0, v0, m, q, r)

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
