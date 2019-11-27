from .checking import check_numerical_return_array
from .checking import check_ndim_return_array
from .checking import check_type_return_list
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
    a = check_ndim_return_array(a, 1)
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
    a = check_ndim_return_array(a, 1)
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in velocity vectors.")
        raise ValueError(msg)
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

            It must be a 2-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 2)
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in position vectors.")
        raise ValueError(msg)
    return a

def validate_velocities(a):
    """
        Checks that the array 'a' is a valid set of velocity vectors, meaning:

            It must be a 2-D array of numbers
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 2)
    if not np.all(np.isfinite(a)):
        msg = (f"Sequence contains non-finite values; finite values expected "
               f"in velocity vectors.")
        raise ValueError(msg)
    return a

def validate_masses(a):
    """
        Checks that the array 'a' is a valid set of masses, meaning:

            It must be a 1-D array of numbers greater than zero
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 1)
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

            It must be a 1-D array
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 1)
    if not np.all(np.isfinite(a)):
        msg = "Non-finite element; finite elements expected for charge"
        raise ValueError(msg)
    return a

def validate_radii(a):
    """
        Checks that the array 'a' is a valid set of radii, meaning:

            It must be a 1-D array of numbers greater than zero
            It must only contain finite and defined values

        If successful, returns 'a' with the given changes
    """
    a = np.array(a)
    a = check_numerical_return_array(a)
    a = check_ndim_return_array(a, 1)
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
