import os
import ctypes
import tempfile
import subprocess

__all__ = [
    "compile_runtime",
]

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
            ctypes.c_int,
        ],
        "restypes": None,
    },
        
}

def compile_runtime(out_dll):
    src_file = f"{os.path.dirname(__file__)}/src/runtime.c"
    cmd = ["cc", "--shared", "-O2", "-o", out_dll, src_file]
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
        _runtime_entries[name].restypes = func_info["restypes"]
    
    
def __getattr__(x):
    _resolve_runtime()
    return _runtime_entries[x]
