from .exceptions import ShapeError
import numpy as np

def check_type_return_list(obj, obj_type):
    """
        [1] Accepts a sequence of object instances, or a single instance.
        [2] If obj is a single instance, checks it is of type 'obj_type' and
            returns this as the single element list.
        [3] If obj is a sequence of instances, checks that all elements are of
            type 'obj_type' and returns the sequence as a list, if successful.
    """
    if isinstance(obj, obj_type):
        obj = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for n,i in enumerate(obj):
            if not isinstance(i, obj_type):
                msg = (f"Element {n:d} is of type {str(type(i))}; all elements"
                       f" must be {str(obj_type)} instances.")
                raise TypeError(msg)
    else:
        msg = (f"Argument is of type {str(type(obj))}; expected an "
               f"instance of type {str(obj_type)} or sequence containing "
               f"such instances.")
        raise TypeError(msg)

    return list(obj)

def check_ndim_return_array(a, ndim):
    """
        Checks that the given sequence 'a' is of the given dimension. Returns
        the array, if successful.
    """
    a = np.array(a)
    if a.ndim != ndim:
        msg = (f"The given sequence is {a.ndim:d}-D; a {ndim:d}-D sequence "
               f"is required.")
        raise ShapeError(msg)
    return a

def check_numerical_return_array(a):
    """
        Checks that the array 'a' contains numbers by converting it to float64,
        then returns it, if it's numerical.
    """
    a = np.array(a)
    try:
        a = a.astype(np.float64)
    except TypeError:
        msg = (f"Sequence contains invalid elements: numerical sequence "
               f"of real numbers expected.")
        raise TypeError(msg)
    except ValueError:
        msg = (f"Sequence contains invalid elements: numerical sequence "
               f"of real numbers expected.")
        raise ValueError(msg)
    return a
