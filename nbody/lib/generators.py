from ..utils.validation import validate_velocities
from ..utils.validation import validate_positions
from ..utils.validation import validate_charges
from ..utils.validation import validate_masses
from ..utils.validation import validate_radii
from ..utils.validation import init_parser
from ..core import System
import numpy as np

def spheres(*args, **kwargs):
    """

        Returns a System instance containing the initial conditions passed to
        this function.

        All arguments must be numerical arrays, and all values must be in
        S.I. units.

        –– INPUT ARGUMENTS ––––––––––––––––––––––––––––––––––––––––––––––––

                ARG     DESCRIPTION     SHAPE       S.I. UNITS

                x0  –   positions   –   (N,p)   –   meters
                v0  –   velocities  –   (N,p)   –   meters/second
                w0  –   ang. vel.   –   (N,p)   –   radians/second
                m   –   masses      –   (N,)    –   kilograms
                q   –   charges     –   (N,)    –   coulombs
                r   –   radii       –   (N,)    –   meters

    """
    S = System()

    # Checking each argument is programmatically and physically sound
    parsed_args = init_parser(*args, **kwargs)

    S.x0 = parsed_args["x0"]
    S.v0 = parsed_args["v0"]
    S.w0 = parsed_args["w0"]
    S.m = parsed_args["m"]
    S.q = parsed_args["q"]
    S.r = parsed_args["r"]

    # Initializing dimensionality
    S.N = S.x0.shape[0]
    S.p = S.x0.shape[1]

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

    # Setting lattice masses
    m = np.ones((N,1))*mass

    # Setting lattice charges
    bool_shape = ((shape + 1) % 2).astype(bool)
    slices = [slice(i) for i in shape] if shape.ndim > 0 else slice(shape)
    shape[bool_shape] += 1
    q = np.ones(np.prod(shape))
    q[::2] = -1
    q = q.reshape(shape)
    q = q[tuple(slices)]*charge
    q = q.reshape((N,1))

    # Setting lattice radii
    r = np.ones((N,1))*radius

    return spheres(x0 = x0, m = m, q = q, r = r)

def rand(N, p = 2, x0 = (0,100), v0 = (0,100), w0 = (0,1), m = (1E7,1E5), q = (0,1E-5), r = (1,0.1)):
    N = int(N)
    x0 = np.random.normal(x0[0], x0[1], (N,p))
    v0 = np.random.normal(v0[0], v0[1], (N,p))
    if p == 2:
        w0 = np.random.normal(w0[0], w0[1], (N,1))
    elif p == 3:
        w0 = np.random.normal(w0[0], w0[1], (N,3))
    m = np.random.normal(m[0], m[1], (N, 1))
    m[m < 0] = np.abs(m[m < 0])
    m[m == 0] = 1
    q = np.random.normal(q[0], q[1], (N, 1))
    r = np.random.normal(r[0], r[1], (N, 1))
    r[r < 0] = np.abs(r[r < 0])
    r[r == 0] = 1
    if p in [2,3]:
        return spheres(x0, v0, w0, m, q, r)
    else:
        return spheres(x0 = x0, v0 = v0, m = m, q = q, r = r)

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
