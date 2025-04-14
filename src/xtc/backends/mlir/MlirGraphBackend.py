from typing import cast, Any
from typing_extensions import override

from xdsl.dialects.func import FuncOp as xdslFuncOp
from xdsl.dialects.builtin import AnyMemRefType, MemRefType, f32, f64
from xdsl.ir import Block as xdslBlock
from xdsl.ir import Region as xdslRegion

from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph
from xtc.graphs.xtc.data import XTCTensorType
from xtc.graphs.xtc.expr import XTCTensorExpr

from .MlirNodeBackend import MlirNodeBackend
from .MlirBackend import MlirBackend
from .MlirOps import MlirOperation


class MlirGraphBackend(MlirBackend):

    def __init__(
        self,
        xdsl_func: xdslFuncOp|Graph,
        nodes: list[MlirNodeBackend]|None = None,
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
        no_alias: bool = False,
    ):
        if isinstance(xdsl_func, XTCGraph):
            assert nodes is None
            graph = xdsl_func
            function, nodes_dict = self._init_from_graph(graph)
        else:
             assert isinstance(xdsl_func, xdslFuncOp)
             assert nodes is not None
             graph = None
             function, nodes_dict = self._init_from_xdsl(xdsl_func, nodes)
        self.nodes = nodes_dict
        super().__init__(
            xdsl_func = function,
            always_vectorize = always_vectorize,
            concluding_passes = concluding_passes,
            no_alias = no_alias,
            graph = graph,
        )

    def _init_from_xdsl(
            self,
            function: xdslFuncOp,
            nodes: list[MlirNodeBackend],
    )  -> tuple[xdslFuncOp, dict[str, MlirNodeBackend]]:
        nodes_dict = {}
        for impl in nodes:
            first_block = cast(xdslBlock, function.body.first_block)
            assert impl.source_op in first_block.ops
            nodes_dict[impl.payload_name] = impl
        return function, nodes_dict

    def _init_from_graph(
            self,
            graph: XTCGraph,
            concluding_passes: list[str] = [],
            always_vectorize: bool = True,
            no_alias: bool = False,
    ) -> tuple[xdslFuncOp, dict[str, MlirNodeBackend]]:
        operations = [
            MlirOperation.from_operation(node.operation, name=node.name)
            for node in graph.nodes.values()
        ]
        inputs_types = [
            cast(XTCTensorExpr, node._expr).type
            for node in graph.inputs_nodes
        ]
        blocks_and_attrs = [oper.generate() for oper in operations]
        blocks_attrs = [attrs for _, attrs in blocks_and_attrs]
        region = xdslRegion([block for block, _ in blocks_and_attrs])  # type: ignore # mypy issue with dataclass
        ops_types = [
            self._xdsl_type_from_tensortype(input_type)
            for input_type in inputs_types
        ]
        payload = xdslFuncOp.from_region(
            name=graph.name,
            input_types=ops_types,
            return_types=[],
            region=region,
        )
        nodes_dict = {}
        for attrs in blocks_attrs:
            for (node_id, node), dims in (
                    zip(attrs["nodes_map"].items(), attrs["dims_sizes"])
            ):
                nodes_dict[node_id] = MlirNodeBackend(
                    payload_name=node_id,
                    source_op=node,
                    dims=dims,
                    no_alias=no_alias,
                    always_vectorize=always_vectorize,
                    concluding_passes=concluding_passes,
                    id=f"__xtc_id_{node_id}_")
        return payload, nodes_dict

    def _xdsl_type_from_tensortype(self, type: XTCTensorType) -> Any:
        elt_type = {"float32": f32, "float64": f64}[type.constant_dtype]
        return MemRefType(elt_type, type.constant_shape)

    def _np_types_spec(
        self, types: list[AnyMemRefType]
    ) -> list[dict[str, tuple[int, ...] | str]]:
        types_map = {"f32": "float32", "f64": "float64"}
        types_spec: list[dict[str, tuple[int, ...] | str]] = [
            {
                "shape": t.get_shape(),
                "dtype": types_map[str(t.get_element_type())],
            }
            for t in types
        ]
        return types_spec

    @override
    def np_inputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        inputs_args_types = [arg.type for arg in self.xdsl_func.args[:-1]]
        list_memref_tys = cast(list[AnyMemRefType], inputs_args_types)
        return self._np_types_spec(list_memref_tys)

    @override
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        outputs_args_types = [arg.type for arg in self.xdsl_func.args[-1:]]
        list_memref_tys = cast(list[AnyMemRefType], outputs_args_types)
        return self._np_types_spec(list_memref_tys)
