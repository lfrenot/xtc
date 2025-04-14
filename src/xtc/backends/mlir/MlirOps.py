from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any, Type, TypeAlias

from xdsl.dialects import func, linalg, arith, builtin
from xdsl.dialects.builtin import MemRefType, f32, f64, i64, UnitAttr, DenseIntOrFPElementsAttr, StringAttr, AffineMapAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.irdl import irdl_op_definition
from xdsl.builder import ImplicitBuilder

from xtc.itf.graph import Operation


__all__ = [
    "MlirOperation",
    "MlirOperator",
    "MlirOperators",
]

OpAttrs: TypeAlias = dict[str, Any]

class MlirOperation():
    def __init__(
            self,
            operator: Type["MlirOperator"],
            args: tuple[Any,...],
            attrs: dict[str, Any] = {},
            name: str|None = None,
    ) -> None:
        self.operator = operator(args, attrs, name=name)
        self.args = args
        self.attrs = attrs
        self.name = self.operator.name if name is None else name

    def generate(self) -> tuple[Block, OpAttrs]:
        return self.operator.generate_op()

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
    def from_operation(cls, xtc_op: Operation, name: str|None) -> "MlirOperation":
        dims = xtc_op.dims.values()
        dtype = xtc_op.inputs_types[0].dtype # TODO: infer dtype form first input
        args = tuple([*dims, dtype])
        attrs = xtc_op.attrs
        return MlirOperation(
            MlirOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
        )



class MlirOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(self, args: tuple[Any,...], attrs: dict[str, Any], name: str|None = None) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

    @abstractmethod
    def generate_op(self) -> tuple[Block, OpAttrs]:
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


class MlirOperatorMatmul(MlirOperator):
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
    def generate_op(self) -> tuple[Block, OpAttrs]:
        Ki, Kj, Kk, dtype = self.args
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        ops_types = [MemRefType(elt_type, shape)
                     for shape in [[Ki,Kk], [Kk,Kj], [Ki,Kj]]]
        block = Block(arg_types=ops_types)
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            fill = linalg.FillOp(
                res=(),
                inputs=(cst0.results[0],),
                outputs=(block.args[2],),
            )
            reduce = linalg.MatmulOp(
                res=(),
                inputs=(block.args[0], block.args[1]),
                outputs=(block.args[2],),
            )
            func.ReturnOp()
        fill_node_id = f"{self.name}_fill"
        reduce_node_id = f"{self.name}_reduce"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"i": Ki, "j": Kj},
                self.dims_sizes(),
            ],
        }
        return block, attrs

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


@irdl_op_definition
class Conv2DNhwcHwFcOp(linalg.ConvOpsBase):
    """
    Performs 2-D convolution with inputs (N, H, W, C) (R, S, C F)

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nhwc_hwcf-linalgconv2dnhwchwcfop
    """

    name = "linalg.conv_2d_nhwc_hwcf"

class MlirOperatorConv2D(MlirOperator):
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
        sh, sw = self.attrs["stride"]
        inps_dims = self.inputs_dims()
        out_dims = self.outputs_dims()[0]
        dtype = self.args[-1]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        ops_types = [MemRefType(elt_type, shape)
                     for shape in [*inps_dims, out_dims]]
        block = Block(arg_types=ops_types)
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            fill = linalg.FillOp(
                res=(),
                inputs=(cst0.results[0],),
                outputs=(block.args[2],),
            )
            # TODO: Does not work
            # strides = DenseIntOrFPElementsAttr.vector_from_list([sh, sw], i64)
            # dilations = DenseIntOrFPElementsAttr.vector_from_list([1, 1], i64)
            # reduce = Conv2DNhwcHwFcOp(
            #     inputs=(block.args[0], block.args[1]),
            #     outputs=(block.args[2],),
            #     dilations=dilations,
            #     strides=strides,
            # )
            iterator_types = [
                StringAttr({"P": "parallel", "R": "reduction"}[k])
                for k in self.KINDS
            ]
            block_in = Block(arg_types=[f32, f32, f32])
            with ImplicitBuilder(block_in):
                mul = arith.MulfOp(block_in.args[0], block_in.args[1])
                add = arith.AddfOp(block_in.args[2], mul)
                linalg.YieldOp(add)
            reduce = linalg.GenericOp(
                inputs=(block.args[0], block.args[1]),
                outputs=(block.args[2],),
                body=Region([block_in]), # type: ignore # mypy issue with dataclass
                # ignore typing due to xdsl hints limitation
                indexing_maps=[
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: # type: ignore
                            (b, h * sh + r, w * sw + s, c)
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: # type: ignore
                            (r, s, c, f)
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: # type: ignore
                            (b, h, w, f)
                        )
                    ),
                ],
                iterator_types=iterator_types,
            )
            func.ReturnOp()
        fill_node_id = f"{self.name}_fill"
        reduce_node_id = f"{self.name}_reduce"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"b": Kb, "h": Kh, "w": Kw, "f": Kf},
                self.dims_sizes(),
            ],
        }
        return block, attrs

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




class MlirOperators():
    @classmethod
    def from_name(cls, name: str) -> Type[MlirOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = MlirOperatorMatmul
    conv2d = MlirOperatorConv2D
