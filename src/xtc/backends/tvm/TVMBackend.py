import tempfile
from typing import Any
from pathlib import Path
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph

from .TVMOps import TVMOperation
from .TVMScheduler import TVMScheduler
from .TVMCompiler import TVMCompiler

__all__ = [
    "TVMBackend",
]


class TVMBackend(itf.back.Backend):
    def __init__(
            self,
            source_op: TVMOperation|Graph,
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
                TVMOperation.from_operation(node.operation, name=node.name)
                for node in graph.nodes.values()
            ]
            self.dims = self.ops[-1].operator.dims_sizes()
            self.payload_name = self._graph.name
        else:
            assert isinstance(source_op, TVMOperation)
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

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return TVMScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return TVMCompiler(self, **kwargs)

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
