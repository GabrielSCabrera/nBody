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

Using ```numpy``` arrays, you will need:

* An initial position array ```x``` with shape ```(N,p)```
    * *N* is the number of *particles*
    * *p* is the number of *dimensions*
* An initial velocity array ```v``` with shape ```(N,p)```
* An array of masses ```m```with shape ```(N,)```
* An array of charges ```q``` with shape ```(N,)```
* An array of radii ```r``` with shape ```(N,)```

A possible configuration is as follows:

    import numpy as np
    x = np.random.normal(0,   10, (N,p)) # Positions
    v = np.random.normal(0,    2, (N,p)) # Velocities
    m = np.random.normal(8,    1, (N, )) # Masses
    q = np.random.normal(0, 1E-6, (N, )) # Charges
    r = np.random.normal(1,  0.1, (N, )) # Radii

    m[m < 0] = np.abs(m[m < 0])
    m[m == 0] = 1E-3

Next, pass these arrays in the given order to the ```spheres``` function, so as to create a new instance ```S``` of class ```System``` with the above conditions.

    from nbody import *
    S = spheres(x, v, m, q, r)

After selecting a simulation runtime ```T``` and (optional) time-step ```dt```, use the ```solve``` method to calculate the particles' trajectories.

    S.solve(T, dt)

If the system is 2-D such that ```p == 2```, an animation can be created and saved to file; here, the filename ```quick_start``` is chosen, and will produce a file ```animations/quick_start.mp4```.  

    animate(S, "quick_start")

If the system is 3-D such that ```p == 3```, animations can be created but not saved to file – simply omit the string argument shown above, and no warnings will be raised.

Once the ```solve``` method has been called, it is also possible to save the ```System``` instance to file; in this case, the data will be saved to a directory ```saved/quick_start```.

    save(S, "quick_start")
