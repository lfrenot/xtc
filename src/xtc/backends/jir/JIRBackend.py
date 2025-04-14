from typing_extensions import override
from typing import Any
from pathlib import Path
import tempfile

from jir.node import JIRFunction
from jir.backend.util.annotate_fastmath import annotate_fastmath
from jir.parser import JIRParser
from jir.backend.xdsl.compiler import PolygeistCompiler

from xtc.utils.tools import (
    get_geist_prefix,
)

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph

from .JIROps import JIROperation
from .JIRScheduler import JIRScheduler
from .JIRCompiler import JIRCompiler

__all__ = [
    "JIRBackend",
]

class JIRBackend(itf.back.Backend):
    def __init__(
            self,
            source_op: JIROperation|Graph,
            dims: dict[str,int]|None = None,
            parallel_dims: list[str]|None = None,
            reduction_dims: list[str]|None = None,
            **kwargs: Any,
    ) -> None:
        self._graph: Graph|None = None
        if isinstance(source_op, XTCGraph):
            graph = source_op
            self._graph = graph
            self.ops = [
                JIROperation.from_operation(node.operation, name=node.name)
                for node in graph.nodes.values()
            ]
            self.dims = self.ops[-1].operator.dims_sizes()
            self.payload_name = self._graph.name
        else:
            assert isinstance(source_op, JIROperation)
            assert dims is not None
            self.dims = dims
            self.ops = [source_op]
            self.payload_name = source_op.name

        self.op = self.ops[-1]

        assert tuple(self.dims.keys()) == self.op.operator.dims(), (
            f"incompatible dims names: {tuple(self.dims.keys())} != "
            f"{self.op.operator.dims()}"
        )
        self.parallel_dims = self.op.operator.dims("P")
        self.reduction_dims = self.op.operator.dims("R")
        if parallel_dims is not None:
            assert tuple(parallel_dims) == self.parallel_dims, (
                f"incompatible parallel dims names: {tuple(parallel_dims)} != "
                f"{self.parallel_dims}"
            )
        if reduction_dims is not None:
            assert tuple(reduction_dims) == self.reduction_dims, (
                f"incompatible reduction dims names: {tuple(reduction_dims)} != "
                f"{self.reduction_dims}"
            )
        self._geist_install_dir = get_geist_prefix()
        self._op_function_str, self._jir_function_str = self.op.generate(self.payload_name)
        self._jir_function_op = self._parse_function(self._jir_function_str)
        self._op_function_mlir = self._parse_primitives(self._op_function_str)

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return JIRScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return JIRCompiler(self, **kwargs)

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        assert self._graph is not None
        return self._graph

    def evaluate(
            self,
            schedule: itf.schd.Schedule,
            compiler_args: dict = {},
            evaluate_args: dict = {}) -> float|str:
        with tempfile.TemporaryDirectory() as dirname:
            libpath = Path(dirname) / f"payload_{self.payload_name}"
            compiler = self.get_compiler(
                dump_file=str(libpath),
                shared_lib=True,
                **compiler_args,
            )
            module = compiler.compile(schedule)
            evaluator = module.get_evaluator(
                validate=True,
                **evaluate_args,
            )
            results, code, error_msg = evaluator.evaluate()
        return min(results) if code == 0 else error_msg

    def _parse_function(self, jir_function: str) -> JIRFunction:
        return JIRParser().parse_function(jir_function)

    def _parse_primitives(self, op_function: str) -> str:
        polygeist_compiler = PolygeistCompiler(f"{self._geist_install_dir}/bin/cgeist")
        return annotate_fastmath(polygeist_compiler(op_function))
