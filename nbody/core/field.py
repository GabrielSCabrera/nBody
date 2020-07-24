import numpy as np
import itertools

class Field:

    def __init__(self, f):
        """
            Takes a function 'f' that accepts any of the kwargs t, x, v, w, m,
            q, and/or r.  No others are accepted, unless they have default
            values.
        """
        if not callable(f):
            msg = (f"Argument 'f' in 'Field.set_function(f)' must be a "
                   f"function that accepts kwargs t, x, v, w, m, q, and/or r")
            raise TypeError(msg)
        self._get_kwargs(f)
        self.f = f

    def _get_kwargs(self, f):
        """
            Determines which combination of kwargs is accepted in function 'f'
        """
        keys = np.array(["t","x","v","w","m","q","r"])
        f_keys = f.__code__.co_varnames[:f.__code__.co_argcount]
        for key in f_keys:
            if key not in keys:
                msg = (f"Invalid parameter '{key}' found in function "
                       f"'{f.__code__.co_name}'.  Valid parameters are "
                       f"t, x, v, w, m, q, and/or r.")
                raise KeyError(msg)
        self.kwargs = f_keys

    def __call__(self, **kwargs):
        """
            Accepts arguments t, x, v, w, m, q, and/or r – the arguments need
            not be of same shape, but their first axes must all be of equal
            length.

            Must concur with the function set in method '__init__'
        """
        filter_kwargs = {}
        for key, val in kwargs.items():
            if key in self.kwargs:
                filter_kwargs[key] = val
        try:
            return self.f(**filter_kwargs)
        except TypeError as e:
            msg = (f"{e}, when called as Field instance")
            raise TypeError(msg)
