import os,sys

sys.path.append('../')

from MlirImplementer import MlirImplementer

from xdsl.dialects import func,linalg,arith
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    IntegerType,
    i32,
    f32,
    AffineMapAttr,
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region

home = os.environ.get("HOME","")
vectors_size = 16

dim_size = 10

qty = IntegerType(8)

block = Block(arg_types=(qty,f32))

scale = arith.Constant(arith.FloatAttr(0.1, f32))
zero_point = arith.Constant.from_int_and_width(1, 32)
in_f32 = arith.SIToFPOp(block.args[0],f32)
zero_point_f32 = arith.SIToFPOp(zero_point.results[0],f32)
dequantized = arith.Subf(in_f32.results[0],zero_point_f32.results[0])
dequantized_scaled = arith.Mulf(dequantized.results[0],scale.results[0])
block.add_ops([
    scale,
    zero_point,
    in_f32,
    zero_point_f32,
    dequantized,
    dequantized_scaled,
    linalg.YieldOp(dequantized_scaled.results[0]),
])

dequant_generic = linalg.Generic(
    (TestSSAValue(MemRefType(qty, (dim_size,))),),
    (TestSSAValue(MemRefType(f32, (dim_size,))),),
    Region(block),
    (
        AffineMapAttr(AffineMap.identity(1)),
        AffineMapAttr(AffineMap.identity(1)),
    ),
    (linalg.IteratorTypeAttr.parallel(),),
)

impl = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = dequant_generic,
    dims = {'i':dim_size},
    parallel_dims = ['i'],
    reduction_dims = [],
    vectors_size = vectors_size
)

e = impl.evaluate(
    print_source_ir=True,
    print_transformed_ir=False,
    print_lowered_ir = False,
    print_assembly=False,
    color = True,
    debug = False,
)

print(e)
