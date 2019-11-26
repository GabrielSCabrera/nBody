import numpy as np

class Sphere:

    def __init__(self, x0, v0, m, q, r):
        """
        –– INPUT ARGUMENTS ––––––––––––––––––––––––––––––––––––––––––––––––

                ARG     DESCRIPTION     SHAPE       S.I. UNITS

                x0  –   positions   –   (N,p)   –   meters
                v0  –   velocities  –   (N,p)   –   meters/second
                m   –   masses      –   (N,)    –   kilograms
                q   –   charges     –   (N,)    –   coulombs
                r   –   radii       –   (N,)    –   meters
        """
        # Initializing Position, Velocity, Mass, and Radius
        self.x0 = np.array(x0).squeeze()
        self.v0 = np.array(v0).squeeze()
        self.m = np.array(m).squeeze()
        self.q = np.array(q).squeeze()
        self.r = np.array(r).squeeze()

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
