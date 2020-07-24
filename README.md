# nBody

### A GPU-accelerated N-body particle simulator and animator

Create complex particle simulations the easy way: a high-level package for designing and simulating large-scale particle interactions. Let **nBody** do the hard work for you!

## Features

Easy to use – and fast – **nBody** can simulate:

* Gravitational acceleration
* Coulomb interactions
* Particle collisions

**nBody** is highly optimized:

* GPU acceleration available via [```cupy```](https://cupy.chainer.org "cuPY")
* CPU multiprocessing with [```numpy```](https://numpy.org/ "NumPy")
* Energy conservation via the *velocity-verlet* algorithm

Animated [```matplotlib```](https://matplotlib.org/ "Matplotlib") visualizations included for 2-D simulations. 3-D animations are also supported through the use of [```vpython```](https://vpython.org/ "VPython").

## Quick-Start

The package can be installed with the *python package installer*:

    pip3 install nbody

Using ```numpy``` arrays, you will need:

* An initial position array ```x0``` with shape ```(N,p)```
    * *N* is the number of *particles*
    * *p* is the number of *dimensions*
    
All other arguments are optional:

* An initial velocity array ```v0``` with shape ```(N,p)```
* An initial angular velocity array ```w0``` (supported for 2-D and 3-D systems *only*)
    * In 2-D, with shape ```(N,1)``` 
    * In 3-D, with shape ```(N,3)```
* An array of masses ```m```with shape ```(N,1)```
* An array of charges ```q``` with shape ```(N,1)```
* An array of radii ```r``` with shape ```(N,1)```

A possible configuration is as follows:

    import numpy as np
    x0 = np.random.normal(0, 10,   (N,p)) # Positions
    v0 = np.random.normal(0, 2,    (N,p)) # Velocities
    w0 = np.random.normal(0, 1,    (N,1)) # Angular Velocities (not yet implemented)
    m  = np.random.normal(8, 1,    (N,1)) # Masses
    q  = np.random.normal(0, 1E-6, (N,1)) # Charges
    r  = np.random.normal(1, 0.1,  (N,1)) # Radii

    m[m < 0] = np.abs(m[m < 0])
    m[m == 0] = 1E-3
    
    r[r < 0] = np.abs(r[r < 0])
    r[r == 0] = 1E-3

Next, pass these arrays in the given order to the ```spheres``` function, so as to create a new instance ```S``` of class ```System``` with the above conditions.

    import nbody as nb
    S = nb.spheres(x0 = x0, v0 = v0, w0 = w0, m = m, q = q, r = r)

After selecting a simulation runtime ```T``` and (optional) time-step ```dt```, use the ```solve``` method to calculate the particles' trajectories.
    
    T = 1      # Total simulation runtime
    dt = 1E-3  # Simulation time-step
    S.solve(T, dt)

If the system is 2-D such that ```p == 2```, an animation can be created and saved to file; here, the filename ```quick_start``` is chosen, and will produce a file ```animations/quick_start.mp4```.  

    nb.animate(S, "quick_start")

If the system is 3-D such that ```p == 3```, animations can be created but not saved to file – simply omit the string argument shown above, and no warnings will be raised.

Once the ```solve``` method has been called, it is also possible to save the ```System``` instance to file; in this case, the data will be saved to a directory ```saved/quick_start```.

    nb.save(S, "quick_start")
