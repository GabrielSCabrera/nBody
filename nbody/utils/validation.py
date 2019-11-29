from ..config.default_params import default_params
from .checking import check_numerical_return_array
from .checking import check_ndim_return_array
from ..utils.exceptions import DimensionError
from .checking import check_type_return_list
from ..utils.exceptions import PositionError
from ..utils.exceptions import ArgumentError
from .exceptions import ShapeError
import numpy as np

# For individual Spheres

def validate_position(a):
    """
        Checks that the array 'a' is a valid position vector, meaning:

            It must be a scalar, or 1-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, (0,1))
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in position vectors.")
        raise ValueError(msg)
    return a

def validate_velocity(a):
    """
        Checks that the array 'a' is a valid velocity vector, meaning:

            It must be a scalar, or 1-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, (0,1))
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in velocity vectors.")
        raise ValueError(msg)
    return a

def validate_angular_velocity(a, ndim):
    """
        Checks that the array 'a' is a valid angular velocity, meaning:

            FOR THE 2-D CASE
                It must be a scalar
                It must only contain finite and defined values

            FOR THE 3-D CASE
                It must be a 3-vector
                It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)

    if ndim == 2:
        a = check_ndim_return_array(a, 0)
        if not np.all(np.isfinite(a)):
            msg = (f"Non-finite scalar; positive finite value expected for "
                   f"2-D angular velocity scalar.")
            raise ValueError(msg)
    elif ndim == 3:
        a = check_ndim_return_array(a, 1)
        if a.shape[0] != 3:
            msg = (f"Attempt to initialize 3-D angular velocity with array of "
                   f"length {a.shape[0]:d}; expect length '3'")
            raise ShapeError(msg)
        elif not np.all(np.isfinite(a)):
            msg = ("Sequence contains non-finite values; finite values "
                   "expected in angular velocity sequence.")
            raise ValueError(msg)
    else:
        msg = (f"Attempting to initialize angular momentum for a {ndim:d}-D "
               f"Sphere; only 2-D and 3-D Spheres support rotations.")
        raise ShapeError(msg)
    return a

def validate_mass(a):
    """
        Checks that the value 'a' is a valid mass, meaning:

            It must be a scalar greater than zero
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 0)
    if a <= 0:
        msg = "Non-positive scalar; positive finite value expected for mass."
        raise ValueError(msg)
    if not np.isfinite(a):
        msg = "Non-finite scalar; positive finite value expected for mass."
        raise ValueError(msg)
    return a

def validate_charge(a):
    """
        Checks that the value 'a' is a valid charge, meaning:

            It must be a scalar
            It must be finite and defined

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 0)
    if not np.isfinite(a):
        msg = "Non-finite scalar; finite value expected for charge."
        raise ValueError(msg)
    return a

def validate_radius(a):
    """
        Checks that the value 'a' is a valid radius, meaning:

            It must be a scalar greater than zero
            It must be finite and defined

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 0)
    if a <= 0:
        msg = "Non-positive scalar; positive finite value expected for radius."
        raise ValueError(msg)
    if not np.isfinite(a):
        msg = "Non-finite scalar; positive finite value expected for radius."
        raise ValueError(msg)
    return a

# For sequences of Spheres

def validate_positions(a):
    """
        Checks that the array 'a' is a valid set of position vectors, meaning:

            It must be a 1-D or 2-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, (2))
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in position vectors.")
        raise ValueError(msg)
    return a

def validate_velocities(a):
    """
        Checks that the array 'a' is a valid set of velocity vectors, meaning:

            It must be a 1-D or 2-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, (2))
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in velocity vectors.")
        raise ValueError(msg)
    return a

def validate_angular_velocities(a, ndim):
    """
        Checks that the array 'a' is a valid set of angular velocities:

            FOR THE 2-D CASE
                It must be a 1-D array of numbers
                It must only contain finite and defined values

            FOR THE 3-D CASE
                It must be a 2-D array of numbers
                It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)

    if ndim == 2:
        a = check_ndim_return_array(a, 2)
        if a.shape[1] != 1:
            msg = (f"Attempt to initialize 2-D angular velocities with array "
                   f"of shape ({a.shape[0]:d},{a.shape[1]:d}); second axis "
                   f"must be of size 1.")
            raise ShapeError(msg)
        elif not np.all(np.isfinite(a)):
            msg = (f"Sequence contains non-finite values; finite values "
                   f"expected for 2-D angular velocity vector.")
            raise ValueError(msg)
    elif ndim == 3:
        a = check_ndim_return_array(a, 2)
        if a.shape[1] != 3:
            msg = (f"Attempt to initialize 3-D angular velocities with array "
                   f"of shape ({a.shape[0]:d},{a.shape[1]:d}); second axis "
                   f"must be of size 3.")
            raise ShapeError(msg)
        elif not np.all(np.isfinite(a)):
            msg = ("Sequence contains non-finite values; finite values "
                   "expected in angular velocity sequence.")
            raise ValueError(msg)
    else:
        msg = (f"Attempting to initialize angular momentum for a {ndim:d}-D "
               f"Sphere; only 2-D and 3-D Spheres support rotations.")
        raise ShapeError(msg)
    return a

def validate_masses(a):
    """
        Checks that the array 'a' is a valid set of masses, meaning:

            It must be a 2-D array of numbers greater than zero
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 2)
    if not np.all(np.greater(a, 0)):
        msg = (f"Non-positive element; positive finite elements expected for "
               f"mass.")
        raise ValueError(msg)
    if not np.all(np.isfinite(a)):
        msg = (f"Non-finite element; positive finite elements expected for "
               f"mass.")
        raise ValueError(msg)
    return a

def validate_charges(a):
    """
        Checks that the array 'a' is a valid set of charges, meaning:

            It must be a 2-D array
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 2)
    if not np.all(np.isfinite(a)):
        msg = "Non-finite element; finite elements expected for charge"
        raise ValueError(msg)
    return a

def validate_radii(a):
    """
        Checks that the array 'a' is a valid set of radii, meaning:

            It must be a 2-D array of numbers greater than zero
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 2)
    if not np.all(np.greater(a, 0)):
        msg = (f"Non-positive element; positive finite elements expected for "
               f"radius.")
        raise ValueError(msg)
    if not np.all(np.isfinite(a)):
        msg = (f"Non-finite element; positive finite elements expected for "
               f"radius.")
        raise ValueError(msg)
    return a

# Misc

def validate_time(a):
    """
        Checks that the value 'a' is a valid time, meaning:

            It must be a scalar greater than zero
            It must be finite and defined

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 0)
    if a <= 0:
        msg = "Non-positive scalar; positive finite value expected for time."
        raise ValueError(msg)
    if not np.isfinite(a):
        msg = "Non-finite scalar; positive finite value expected for time."
        raise ValueError(msg)
    return a

def init_parser(*args, **kwargs):
    """
        Used to initialize a 'Sphere' instance, or a sequence of spheres

        OPTION 1
        Accepts *args and **kwargs, in the following order:

            x0  v0  w0  m   q   r

        Where 'N' is the number of spheres, and 'p' is the spheres'
        dimensionalities (e.g. p=3 if we have a 3-D system).  Note that all
        sequences must exclusively contain real, finite numbers.

            ARG     DESCRIPTION                 SHAPE       S.I. UNITS

            x0      Initial positions           (N,p)       meters
            v0      Initial linear velocities   (N,p)       meters/second
            m       Masses                      (N,)        kilograms
            q       Charges                     (N,)        coulombs
            r       Radii                       (N,)        meters

        Angular velocity is a special case that can only be used in 2-D and 3-D

            IN 2-D
            w0      Initial angular velocities  (N,1)       radians/second

            IN 3-D
            w0      Initial angular velocities  (N,p)       radians/second

        The only required argument is 'x0'; the others are all optional, and
        default to the values in '/nbody/config/default_params.py'
    """
    docstr = "\n\nHOW-TO INITIALIZE OBJECTS IN NBODY\n\n\t"
    docstr = docstr + "\n".join(init_parser.__doc__.splitlines()[2:]).strip()

    # Determining if option 1 or option 2 was selected
    # If neither, will raise an exception

    order = ["x0", "v0", "w0", "m", "q", "r"]
    options = {}
    for o in order:
        options[o] = False

    if len(args) == 0 and "x0" not in kwargs:
        msg = f"\n\nAttempt to initialize Sphere without position 'x0'{docstr}"
        raise PositionError(msg)

    validators = [validate_positions, validate_velocities,
                  validate_angular_velocities, validate_masses,
                  validate_charges, validate_radii]

    N = None
    p = None

    for n,(i,j,k) in enumerate(zip(args, order, validators)):
        arg = np.array(i)

        if arg.ndim == 0:
            N_step = 1
            p_step = 1
            arg = np.array([[arg]])
        elif arg.ndim == 1:
            N_step = 1
            p_step = arg.shape[0]
            arg = arg[np.newaxis,:]
        elif arg.ndim == 2:
            N_step = arg.shape[0]
            p_step = arg.shape[1]
        else:
            msg = (f"Positional argument {j} must be a scalar, 1-D array, or "
                   f"2-D array{docstr}")
            raise ShapeError(msg)

        if N is None:
            N = N_step
            p = p_step
        else:
            if N != N_step:
                raise DimensionError(N, N_step)
            elif p != p_step and n < 2:
                raise DimensionError(p, p_step)
            elif p_step != 1 and n > 2:
                raise DimensionError(1, p_step)

        if n != 2:
            options[j] = k(arg)
        else:
            options[j] = k(arg, p)

    for n,key in enumerate(order):

        if key not in options:
            msg = f"Invalid key '{key}' passed to initialize Sphere"
            raise KeyError(msg)
        elif key not in kwargs.keys():
            continue

        val = kwargs[key]

        if options[key] is not False:
            msg = (f"Attempting to pass '{key}' as a positional argument and "
                   f"keyword argument{docstr}")
            raise ArgumentError(msg)

        val = np.array(val)

        if val.ndim == 0:
            N_step = 1
            p_step = 1
            val = np.array([[val]])
        elif val.ndim == 1:
            N_step = 1
            p_step = val.shape[0]
            val = val[np.newaxis,:]
        elif val.ndim == 2:
            N_step = val.shape[0]
            p_step = val.shape[1]
        else:
            msg = (f"Keyword argument {key} must be a scalar, 1-D array, or "
                   f"2-D array{docstr}")
            raise ShapeError(msg)

        if N is None:
            N = N_step
            p = p_step
        else:
            if N != N_step:
                raise DimensionError(N, N_step)
            elif p != p_step and key in ["x0", "v0"]:
                raise DimensionError(p, p_step)
            elif p_step != 1 and key not in ["x0", "v0"]:
                raise DimensionError(1, p_step)


        if key != "w0":
            options[key] = validators[n](val)
        else:
            options[key] = validators[n](val, p)

    for key, val in options.items():
        if val is False:
            if key == "v0":
                options[key] = np.ones_like(options["x0"])*default_params[key]
            elif key == "w0":
                if N == 1:
                    if p == 2:
                        options[key] = np.array([[default_params[key]]])
                    elif p == 3:
                        options[key] = np.ones((1,3))*default_params[key]
                    else:
                        options[key] = None
                else:
                    if p == 2:
                        options[key] = np.ones((N,1))*default_params[key]
                    elif p == 3:
                        options[key] = np.ones((N,3))*default_params[key]
                    else:
                        options[key] = None
            else:
                if N == 1:
                    options[key] = np.array([[default_params[key]]])
                else:
                    options[key] = np.ones((N,1))*default_params[key]

    return options
