from ..utils.validation import validate_position
from ..utils.validation import validate_velocity
from ..utils.validation import validate_charge
from ..utils.validation import validate_radius
from ..utils.validation import validate_mass
from ..utils.validation import init_parser
import numpy as np

class Sphere:

    def __init__(self, *args, **kwargs):
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
        parsed_args = init_parser(*args, **kwargs)

        self.x0 = parsed_args["x0"]
        self.v0 = parsed_args["v0"]
        self.w0 = parsed_args["w0"]
        self.m = parsed_args["m"]
        self.q = parsed_args["q"]
        self.r = parsed_args["r"]

        # Initializing dimensionality
        self.p = self.x0.shape[0]

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
