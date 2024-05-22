import numpy as np
import runtime
import ctypes

__all__ = [
    "Evaluator",
    "Executor",
]

class Evaluator():
    def __init__(self, f, repeat = 1, number = 1, min_repeat_ms = 0):
        assert repeat > 0
        assert number > 0
        assert min_repeat_ms >= 0
        self.repeat = repeat
        self.number = number
        self.min_repeat_ms = min_repeat_ms
        self.func = f
        self.eval = runtime.evaluate

    def __call__(self, *args):
        data_args = [arg.ctypes.data_as(ctypes.c_voidp) for arg in args]
        argsp = (ctypes.c_voidp * len(data_args))(*data_args)
        results = (ctypes.c_double * self.repeat)()
        self.eval(ctypes.cast(results, ctypes.POINTER(ctypes.c_double)),
                  ctypes.c_int(self.repeat),
                  ctypes.c_int(self.number),
                  ctypes.c_int(self.min_repeat_ms),
                  ctypes.cast(self.func, ctypes.CFUNCTYPE(ctypes.c_voidp)),
                  ctypes.cast(argsp, ctypes.POINTER(ctypes.c_voidp)),
                  ctypes.c_int(len(data_args)))
        return np.array(results, dtype="float64")


class Executor():
    def __init__(self, f):
        self.func = f

    def __call__(self, *args):
        data_args = [arg.ctypes.data_as(ctypes.c_voidp) for arg in args]
        self.func(*data_args)
