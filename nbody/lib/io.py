from ..core import System

import numpy as np
import os

def save(system, dirname = "nBody_save_"):
    # Create a folder in which to save files
    if not os.path.isdir("saved"):
        os.mkdir("saved")
    dirname = f"saved/{dirname}"
    # If dirname ends in "_", will automatically number the save directory
    if dirname[-1] == "_":
        ID = 0.0
        # Iterating through files until an unused number is found
        while True:
            ID_text = f"{ID:03.0f}"
            if os.path.isdir(dirname + ID_text):
                # If current number is used, increments to check next one
                ID += 1
            else:
                # If current number is available, creates save directories
                dirname = dirname + ID_text
                os.mkdir(dirname)
                os.mkdir(dirname + "/arr")
                break
    elif not os.path.isdir(dirname):
        # If unnumbered, creates save directory and array subdirectory
        os.mkdir(dirname)
        os.mkdir(dirname + "/arr")
    else:
        # If unnumbered, and directory exists, creates array subdirectory
        if not os.path.isdir(dirname + "/arr"):
            os.mkdir(dirname + "/arr")

    # Saving the instance attributes to .npy files
    np.save(f"{dirname}/arr/t", system.t)
    np.save(f"{dirname}/arr/x", system.x)
    np.save(f"{dirname}/arr/v", system.v)
    np.save(f"{dirname}/arr/w", system.w)
    np.save(f"{dirname}/arr/m", system.m)
    np.save(f"{dirname}/arr/q", system.q)
    np.save(f"{dirname}/arr/r", system.r)

    # Saving metadata, such as time steps, simulation runtime, etc...
    with open(f"{dirname}/metadata.dat", "w+") as outfile:
        msg = (f"dt={system.dt} T={system.T} GPU={system.GPU_active} "
               f"col={system.collision}")
        outfile.write(msg)

    # Creates a human-readable log with info on simulation parameters
    with open(f"{dirname}/log.txt", "w+") as outfile:
        outfile.write(system.simulation_info())

def load(dirname):

    t = np.load(f"{dirname}/arr/t.npy")
    x = np.load(f"{dirname}/arr/x.npy")
    v = np.load(f"{dirname}/arr/v.npy")
    w = np.load(f"{dirname}/arr/w.npy")
    m = np.load(f"{dirname}/arr/m.npy")
    q = np.load(f"{dirname}/arr/q.npy")
    r = np.load(f"{dirname}/arr/r.npy")

    I = Simulation(x[0], v[0], w[0], m, q, r)
    I.t, I.x, I.v, I.w = t, x, v, w

    with open(f"{dirname}/metadata.dat") as infile:
        data = infile.read().split(" ")
        I.dt = float(data[0].split("=")[1])
        I.T = float(data[1].split("=")[1])
        I.GPU = bool(data[2].split("=")[1])
        I.collision = bool(data[3].split("=")[1])

    return I

if __name__ == "__main__":

    msg = ("To see an example of this program, run the files in the 'samples'"
           " directory, or take a look at the README.")
    print(msg)
