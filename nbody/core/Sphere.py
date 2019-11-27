from ..utils.validation import validate_position
from ..utils.validation import validate_velocity
from ..utils.validation import validate_charge
from ..utils.validation import validate_radius
from ..utils.validation import validate_mass
import numpy as np

class Sphere:

    def __init__(self, x0, v0, w0, m, q, r):
        """
        –– INPUT ARGUMENTS ––––––––––––––––––––––––––––––––––––––––––––––––

                ARG     DESCRIPTION     SHAPE       S.I. UNITS

                x0  –   positions   –   (N,p)   –   meters
                v0  –   velocities  –   (N,p)   –   meters/second
                w0  –   ang. vel.   –   (N,p)   –   radians/second
                m   –   masses      –   (N,)    –   kilograms
                q   –   charges     –   (N,)    –   coulombs
                r   –   radii       –   (N,)    –   meters
        """

        # Checking each argument is programmatically and physically sound
        self.x0 = validate_position(x0)
        self.v0 = validate_velocity(v0)
        self.w0 = validate_velocity(w0)
        self.m = validate_mass(m)
        self.q = validate_charge(q)
        self.r = validate_radius(r)

        # Initializing dimensionality
        self.p = self.x0.shape[0]

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
