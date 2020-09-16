class BaseTarget:
    """
    Root BaseTarget class has been created to allow for setting the hierarchy of different targets.
    Classes that inherit from BaseTarget can be used either as outputs of an algorithm or as ground
    truth annotations, but there is no guarantee that this is always possible, i.e. that both options are possible.

    Classes that are only used either for ground truth annotations or algorithm outputs must inherit this class.
    """
    def __init__(self):
        pass


class Target(BaseTarget):
    """
    Classes inheriting from the Target class always guarantee that they can be used for both cases, outputs and
    ground truth annotations.
    Therefore, classes that are only used to provide ground truth annotations
    must inherit from BaseTarget instead of Target. To allow representing different types of
    targets, this class serves as the basis for the more specialized forms of targets.
    All the classes should implement the corresponding setter/getter functions to ensure that the necessary
    type checking is performed (if there is no other technical obstacle to this, e.g., negative performance impact).
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.confidence = None
        self.action = None
