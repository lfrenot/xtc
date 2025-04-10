from abc import ABC, abstractmethod
from typing_extensions import override
import numpy as np
from typing import Any, Type

from xtc.itf.graph import Operation

__all__ = [
    "TVMOperation",
    "TVMOperator",
    "TVMOperators",
]

import tvm
import tvm.te as te

def get_tvm_native_target_options() -> str:
    """
    Returm the tvm target options to pass to llvm.
    """
    from cpuinfo import get_cpu_info
    info = get_cpu_info()
    arch = info["arch_string_raw"]
    flags = info["flags"]
    cpu, attrs, triple = "", "", ""
    if arch == "x86_64":
        triple = "x86_64-linux-gnu"
        if "avx512f" in flags:
            cpu = "skylake-avx512"
        elif "avx2" in flags:
            cpu = "core-avx2"
    elif arch == "aarch64":
        triple = "aarch64-linux-gnu"
        if "asimd" in flags:
            cpu = "cortex-a72"
            attrs = "+neon"
    target_options = []
    if triple:
        target_options.append(f"-mtriple={triple}")
    if cpu:
        target_options.append(f"-mcpu={cpu}")
    if attrs:
        target_options.append(f"-mattrs={attrs}")
    return " ".join(target_options)

def tvm_build_crt(sch: Any, operation: Any, target: str, name: str|None = None) -> Any:
    # We use system-lib with crt runtime such that DSO loading works
    # The generated .so can then be used:
    # - for static compilation as soon as the tvm runtime is provided
    # - for dynamic loading from python
    # Recent version of tvm (i.e. 0.19) have a Runtime object
    # Older version (i.e. 0.16) support passing runtime options in target
    try:
        from tvm.relay.backend import Runtime
        runtime_kwargs = {
            "runtime": Runtime(
                "crt",
                {"system-lib": True}
            ),
        }
    except:
        runtime_kwargs = {}
        target = f"{target} --system-lib --runtime=c"
    return tvm.build(
        sch,
        operation,
        target=target,
        name=name,
        **runtime_kwargs
    )

class TVMOperation():
    def __init__(
            self,
            operator: Type["TVMOperator"],
            args: tuple[Any,...],
            attrs: dict[str, Any] = {},
            name: str|None = None,
            target: str|None = None
    ) -> None:
        self.operator = operator(args, attrs, name=name)
        self.args = args
        self.attrs = attrs
        if target is not None:
            self.target_options = ""
            self.target = target
        else:
            self.target_options = get_tvm_native_target_options()
            self.target = "llvm"
        self.tgt = tvm.target.Target(target=f"{self.target} {self.target_options}")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.name = self.operator.name if name is None else name

    def generate(self) -> Any:
        return self.operator.generate_op()

    def schedule(self, operation: Any, schedule: str = "") -> Any:
        sch = te.create_schedule(operation[-1].op)
        if schedule:
            exec(schedule,
                 {"sch": sch, "obj": operation},
                 {})
        return sch

    def build(self, operation: Any, sch: Any, func_name: str|None = None) -> Any:
        if func_name is None:
            func_name = self.name
        return tvm_build_crt(
            sch,
            operation,
            name=func_name,
            target=self.tgt,
        )

    def lower(self, operation: Any, sch: Any) -> str:
        return tvm.lower(sch, operation, simple_mode=True)

    def np_inputs_spec(self) -> list[dict[str, Any]]:
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype
            in zip(self.operator.inputs_dims(), self.operator.inputs_types())
        ]
        return inputs_spec

    def np_outputs_spec(self) -> list[dict[str, Any]]:
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype
            in zip(self.operator.outputs_dims(), self.operator.outputs_types())
        ]
        return outputs_spec

    @classmethod
    def from_operation(cls, xtc_op: Operation, name: str|None) -> "TVMOperation":
        dims = xtc_op.dims.values()
        dtype = xtc_op.inputs_types[0].dtype # TODO: infer dtype form first input
        args = tuple([*dims, dtype])
        attrs = xtc_op.attrs
        return TVMOperation(
            TVMOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
        )


class TVMOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(self, args: tuple[Any,...], attrs: dict[str, Any], name: str|None = None) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

    @abstractmethod
    def generate_op(self):
        ...
    @abstractmethod
    def dims(self, kind: str = "") -> tuple[str,...]:
        ...
    @abstractmethod
    def dims_sizes(self) -> dict[str, int]:
        ...
    @abstractmethod
    def inputs_dims(self) -> tuple[tuple[int,...],...]:
        ...
    @abstractmethod
    def inputs_types(self) -> tuple[str,...]:
        ...
    @abstractmethod
    def outputs_dims(self) -> tuple[tuple[int,...],...]:
        ...
    @abstractmethod
    def outputs_types(self) -> tuple[str,...]:
        ...

    def _dims(self, kind: str = "") -> tuple[str,...]:
        if kind == "":
            return tuple(self.AXES)
        return tuple([
            a for a, k in zip(self.AXES, self.KINDS)
            if k == kind
        ])

class TVMOperatorMatmul(TVMOperator):
    DEFAULT_NAME = "matmul"
    AXES = "ijk"
    KINDS= "PPR"

    @override
    def dims(self, kind: str = "") -> tuple[str,...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, j, k, _ = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(self) -> Any:
        Ki, Kj, Kk, dtype = self.args
        A = te.placeholder((Ki, Kk), name="A", dtype=dtype)
        B = te.placeholder((Kk, Kj), name="B", dtype=dtype)
        k = te.reduce_axis((0, Kk), "k")
        O = te.compute(
            (Ki, Kj),
            lambda i, j: (
                te.sum(
                    A[i, k] * B[k, j],
                    axis=k,
                )
            ),
            name=self.name,
        )
        return A, B, O

    @override
    def inputs_dims(self) -> tuple[tuple[int,...],...]:
        i, j, k, _ = self.args
        return (i, k), (k, j)

    @override
    def inputs_types(self) -> tuple[str,...]:
        dtype = self.args[-1]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int,...],...]:
        i, j = self.args[:2]
        return (i, j),

    @override
    def outputs_types(self) -> tuple[str,...]:
        dtype = self.args[-1]
        return dtype,


class TVMOperatorRelu(TVMOperator):
    DEFAULT_NAME = "relu"
    DEFAULT_THRESHOLD = 0
    AXES = "i"
    KINDS= "P"

    def __init__(self, args: tuple[Any,...], attrs: dict[str, Any], name: str|None = None) -> None:
        attrs = {
            "threshold": self.DEFAULT_THRESHOLD,
            **attrs
        }
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str,...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, _ = self.args
        return {"i": i}

    @override
    def generate_op(self) -> Any:
        Ki, dtype = self.args
        A = te.placeholder((Ki,), name="A", dtype=dtype)
        O = te.compute(
            (Ki,),
            lambda i,: tvm.tir.max(self.attrs["threshold"], A[i]),
            name=self.name,
        )
        return A, O

    @override
    def inputs_dims(self) -> tuple[tuple[int,...],...]:
        i, _ = self.args
        return (i,),

    @override
    def inputs_types(self) -> tuple[str,...]:
        _, dtype = self.args
        return dtype,

    @override
    def outputs_dims(self) -> tuple[tuple[int,...],...]:
        i, _ = self.args
        return (i,),

    @override
    def outputs_types(self) -> tuple[str,...]:
        _, dtype = self.args
        return dtype,


class TVMOperatorConv2D(TVMOperator):
    DEFAULT_NAME = "conv2d"
    AXES = "bhwfrsc"
    KINDS= "PPPPRRR"

    DEFAULT_STRIDE = (1, 1)

    def __init__(self, args: tuple[Any,...], attrs: dict[str, Any], name: str|None = None) -> None:
        attrs = {
            "stride": self.DEFAULT_STRIDE,
            **attrs
        }
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str,...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        b, h, w, f, r, s, c, _ = self.args
        return {"b": b, "h": h, "w": w, "f": f, "r": r, "s": s, "c": c}


    @override
    def generate_op(self) -> Any:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc, dtype = self.args
        inps_dims = self.inputs_dims()
        A = te.placeholder(inps_dims[0], name="A", dtype=dtype)
        W = te.placeholder(inps_dims[1], name="W", dtype=dtype)
        r = te.reduce_axis((0, Kr), "r")
        s = te.reduce_axis((0, Ks), "s")
        c = te.reduce_axis((0, Kc), "c")
        out_dims = self.outputs_dims()[0]
        Ksh, Ksw = self.attrs["stride"]
        O = te.compute(
            out_dims,
            lambda b, h, w, f: (
                te.sum(
                    A[b, h * Ksh + r, w * Ksw + s, c] * W[r, s, c, f],
                    axis=(r, s, c),
                )
            ),
            name=self.name,
        )
        return A, W, O

    @override
    def inputs_dims(self) -> tuple[tuple[int,...],...]:
        b, h, w, f, r, s, c, _ = self.args
        sh, sw = self.attrs["stride"]
        return ((b, h * sh + r - 1, w * sw + s - 1, c), (r, s, c, f))

    @override
    def inputs_types(self) -> tuple[str,...]:
        dtype = self.args[-1]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int,...],...]:
        b, h, w, f = self.args[:4]
        return (b, h, w, f),

    @override
    def outputs_types(self) -> tuple[str,...]:
        dtype = self.args[-1]
        return dtype,


class TVMOperators():
    @classmethod
    def from_name(cls, name: str) -> Type[TVMOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = TVMOperatorMatmul
    relu = TVMOperatorRelu
    conv2d = TVMOperatorConv2D
