import os
import ctypes
import tempfile
import subprocess

__all__ = [
    "compile_runtime",
]

from runtime_types import DLDevice, DLDataType

_runtime_funcs = {
    "evaluate": {
        "sym": "evaluate",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
        ],
        "restype": None,
    },
    "evaluate_packed": {
        "sym": "evaluate_packed",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ],
        "restype": None,
    },
    "cndarray_new": {
        "sym": "CNDArray_new",
        "argtypes": [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            DLDataType,
            DLDevice,
        ],
        "restype": ctypes.c_voidp,
    },
    "cndarray_del": {
        "sym": "CNDArray_del",
        "argtypes": [
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_copy_from_data": {
        "sym": "CNDArray_copy_from_data",
        "argtypes": [
            ctypes.c_voidp,
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_copy_to_data": {
        "sym": "CNDArray_copy_to_data",
        "argtypes": [
            ctypes.c_voidp,
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_set_alloc_alignment": {
        "sym": "CNDArray_set_alloc_alignment",
        "argtypes": [
            ctypes.c_int64,
        ],
        "restype": None,
    },
}

def compile_runtime(out_dll):
    debug = False # True for verbose
    debug_opts = ["-DRUNTIME_DEBUG=1"] if debug else []
    files = ["evaluate.c", "cndarray.c", "alloc.c"]
    src_dir = f"{os.path.dirname(__file__)}/src"
    src_files = [f"{src_dir}/{file}" for file in files]
    cmd = ["cc", "--shared", "-O2", "-fPIC", "-I", src_dir, *debug_opts, "-o", out_dll, *src_files]
    p = subprocess.run(cmd, text=True)
    assert p.returncode == 0, f"unable to compile runtime: {' '.join(cmd)}"

_runtime_lib = None
def _compile():
    global _runtime_lib
    if _runtime_lib is not None:
        return
    with tempfile.TemporaryDirectory() as tdir:
        lib_path = f"{tdir}/runtime.so"
        compile_runtime(lib_path)
        _runtime_lib = ctypes.CDLL(lib_path)

_runtime_entries = None
def _resolve_runtime():
    global _runtime_entries
    if _runtime_entries is not None:
        return
    _compile()
    _runtime_entries = {}
    for name, func_info in _runtime_funcs.items():
        _runtime_entries[name] = getattr(_runtime_lib, func_info["sym"])
        _runtime_entries[name].argtypes = func_info["argtypes"]
        _runtime_entries[name].restype = func_info["restype"]
    
    
def __getattr__(x):
    _resolve_runtime()
    return _runtime_entries[x]
