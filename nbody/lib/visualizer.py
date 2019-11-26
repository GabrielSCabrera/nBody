from matplotlib.animation import FuncAnimation, writers
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os

try:
    import vpython as vp
    vpython_imported = True
except ImportError:
    vpython_imported = False
    warning_msg = ("\033[01mWARNING\033[m: Module \033[03mvpython\033[m is not"
                   " installed on this system. \033[03mvpython\033[m is "
                   "required to enable 3-D animation.")
    print(warning_msg)

def animate(system, savename = None):
    if system.p == 2:
        _animate_2D(system, savename)
    elif system.p == 3:
        # Warns the user if savename is not none
        msg = (f"Cannot save 3-D animations to file – unsupported feature")
        if savename is not None:
            print(msg)
        _animate_3D(system)

def _animate_2D(system, savename = None):

    # Checks if the current simulation is 2-D, raises an error otherwise
    if system.p != 2:
        raise NotImplemented("Can currently only perform 2-D animation")

    # Creating figure and axes instances
    fig, ax = plt.subplots()
    # Making sure the animation aspect ratio is 1:1
    ax.set_aspect("equal")
    # Selecting a reasonable image resolution
    fig.set_size_inches(15, 9)
    # Removing all axes, lines, ticks, and labels
    plt.axis('off')
    # Making the background black
    fig.set_facecolor("k")

    # Selecting how much green will be used for each sphere at each
    # time step.  RGB is used with R = 1 and B = 0.  G is varied between
    # 0 and 1 depending on the sphere's speed in time

    # Calculating sphere speeds
    speeds = np.linalg.norm(system.v, axis = 2)
    # Rescaling the speeds logarithmically
    speeds_scaled = np.log(speeds + np.min(speeds) + 1E-15)
    # Maximum log of shifted speed
    v_min = np.min(speeds_scaled.flatten())
    # Minimum log of shifted speed
    v_max = np.max(speeds_scaled.flatten())
    # Rescaling the speeds in range 0,1 and subtracting from 1
    colors_g = 1-((speeds_scaled - v_min)/(v_max-v_min))

    # Initializing the circles as a list and appending them to the plot
    circles = []
    for i,j,k in zip(system.x[0], system.r, colors_g[0]):
        # Creating a circle with initial position, radius, and RGB color
        circles.append(Circle(tuple(i), j, color = (1,k,0)))
        # Adding the above circle to the plot
        ax.add_artist(circles[-1])

    # Number of standard deviations to focus animation on
    devs = 3

    # Animation initialization function
    def init():
        x0, x1 = system.x[0,:,0], system.x[0,:,1]
        cond_0 = np.abs(x0 - np.mean(x0)) <= devs*np.std(x0)
        cond_1 = np.abs(x1 - np.mean(x1)) <= devs*np.std(x1)
        idx = np.logical_and(cond_0, cond_1)
        # Removing outliers
        x0 = x0[idx]
        x1 = x1[idx]
        # Calculating the limits of x, compensating for sphere radius
        xlim = np.min(x0-system.r[idx]), np.max(x0+system.r[idx])
        # Calculating the limits of y, compensating for sphere radius
        ylim = np.min(x1-system.r[idx]), np.max(x1+system.r[idx])
        # Choosing the largest scale from xlim or ylim
        scale = xlim if xlim[1]-xlim[0] >= ylim[1]-ylim[0] else ylim
        # Taking the difference to calculate the largest scale
        scale = scale[1] - scale[0]
        if xlim[1]-xlim[0] >= ylim[1]-ylim[0]:
            # If x has larger limits, scales y accordingly
            y_mid = (ylim[1] - ylim[0])/2 + ylim[0]
            y_max = y_mid + scale/2
            y_min = y_mid - scale/2
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(y_min, y_max)
        else:
            # If y has larger limits, scales x accordingly
            x_mid = (xlim[1] - xlim[0])/2 + xlim[0]
            x_max = x_mid + scale/2
            x_min = x_mid - scale/2
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(ylim[0], ylim[1])
        return

    # Animation update frame function
    def update(m):
        # Iterating through each circle, for a single step with index m
        for n,c in enumerate(circles):
            # Moving all circles to their new centers
            c.center = tuple(system.x[m,n])
            # Adjusting the green setting depending on current speed
            c.set_color((1,colors_g[m,n],0))

        x0, x1 = system.x[m,:,0], system.x[m,:,1]
        cond_0 = np.abs(x0 - np.mean(x0)) <= devs*np.std(x0)
        cond_1 = np.abs(x1 - np.mean(x1)) <= devs*np.std(x1)
        idx = np.logical_and(cond_0, cond_1)
        # Removing outliers
        x0 = x0[idx]
        x1 = x1[idx]

        # Calculating the limits of x, compensating for sphere radius
        xlim = (np.min(x0-system.r[idx]), np.max(x0+system.r[idx]))
        # Calculating the limits of y, compensating for sphere radius
        ylim = (np.min(x1-system.r[idx]), np.max(x1+system.r[idx]))
        # Choosing the largest scale from xlim or ylim
        scale = xlim if xlim[1]-xlim[0] >= ylim[1]-ylim[0] else ylim
        # Taking the difference to calculate the largest scale
        scale = scale[1] - scale[0]
        if xlim[1]-xlim[0] >= ylim[1]-ylim[0]:
            # If x has larger limits, scales y accordingly
            y_mid = (ylim[1] - ylim[0])/2 + ylim[0]
            y_max = y_mid + scale/2
            y_min = y_mid - scale/2
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(y_min, y_max)
        else:
            # If y has larger limits, scales x accordingly
            x_mid = (xlim[1] - xlim[0])/2 + xlim[0]
            x_max = x_mid + scale/2
            x_min = x_mid - scale/2
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(ylim[0], ylim[1])
        return

    # The frame indices for each integration step
    frames = np.arange(0, system.x.shape[0], 1)

    # Initializing the animator
    anim = FuncAnimation(fig, update, frames = frames, init_func = init,
                        blit = False, interval = 25)

    if savename is None:
        # Display the animation in an interactive session
        plt.show()
    else:
        # Create a folder in which to save files
        if not os.path.isdir("animations"):
            os.mkdir("animations")
        # Setting the video background to black
        savefig_kwargs = {'facecolor':fig.get_facecolor(),
                          'repeat':True}
        # Video metadata
        metadata = {"title":"Particle Simulation",
                    "artist":"Gabriel S Cabrera",
                    "copyright":"GNU General Public License v3.0",
                    "comment":f"Number of spheres: {system.N:d}"}

        # Save the animation to file using ffmpeg
        Writer = writers['ffmpeg']
        writer = Writer(fps = 60, metadata = metadata, bitrate = 2500)
        anim.save(f"animations/{savename}.mp4", writer = writer,
                  savefig_kwargs = savefig_kwargs)
        plt.close()
        # Play the video
        file_path = f"{os.getcwd()}/animations/{savename}.mp4"
        os.system(f'xdg-open {file_path}')

def _animate_3D(system, savename = None):

    # Checks if vpython was successfully imported
    if not vpython_imported:
        raise ImportError("Package 'vpython' required for 3-D animation")

    # Checks if the current simulation is 2-D, raises an error otherwise
    if system.p != 3:
        raise NotImplemented("Can only perform 3-D animation")

    # Selecting how much green will be used for each sphere at each
    # time step.  RGB is used with R = 1 and B = 0.  G is varied between
    # 0 and 1 depending on the sphere's speed in time

    # Calculating sphere speeds
    speeds = np.linalg.norm(system.v, axis = 2)
    # Rescaling the speeds logarithmically
    speeds_scaled = np.log(speeds + np.min(speeds) + 1E-15)
    # Maximum log of shifted speed
    v_min = np.min(speeds_scaled, axis = 1)[:,np.newaxis]
    # Minimum log of shifted speed
    v_max = np.max(speeds_scaled, axis = 1)[:,np.newaxis]
    # Rescaling the speeds in range 0,1 and subtracting from 1
    idx = np.greater(np.abs(v_min - v_max), 1E-10).squeeze()
    colors_g = np.zeros_like(speeds_scaled)
    colors_g[idx] = 1-((speeds_scaled[idx] - v_min[idx])/\
                    (v_max[idx]-v_min[idx]))

    # Initializing the circles as a list and appending them to the plot
    spheres = []
    for i,j,k in zip(system.x[0], system.r, colors_g[0]):
        # Creating a circle with initial position, radius, and RGB color
        pos = vp.vector(i[0], i[1], i[2])
        rgb = vp.vector(1,k,0)
        sphere = vp.sphere(pos = pos, radius = j, color = rgb)
        spheres.append(sphere)

    while True:
        for m in range(system.x.shape[0]):
            vp.rate(32)
            # Iterating through each sphere, for a single step with index m
            for n,s in enumerate(spheres):
                # Moving all circles to their new centers
                s.pos = vp.vector(system.x[m,n,0],system.x[m,n,1],system.x[m,n,2])
                # Adjusting the green setting depending on current speed
                s.color = vp.vector(1 - colors_g[m,n], 0, colors_g[m,n])
