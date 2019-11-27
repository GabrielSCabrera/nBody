class DimensionError(Exception):
    """
        Raised when attempting to combine objects of different dimensions
    """
    def __init__(self, expected, given):
        """
            Arguments 'expected' and 'given' should be of type <int>;
            'expected' is the dimension of the main instance, and 'given' is
            the dimension of the other object being combined with it.
        """

        message = (f"Attempted to combine a {given:d}-D system with a "
                   f"{expected:d}-D system.")

        # Call the base class constructor with the parameters it needs
        super(DimensionError, self).__init__(message)

class ShapeError(Exception):
    """
        Raised when an array is of invalid shape.
    """
    def __init__(self, message):
        super(ShapeError, self).__init__(message)
