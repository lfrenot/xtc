import os,sys

sys.path.append('../')

from MlirImplementer import MlirImplementer
from MlirGraphImplementer import MlirGraphImplementer

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
    FloatAttr,
)
from xdsl.dialects.arith import (
    Constant,
    Mulf,
    Addf,
    FastMathFlagsAttr
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 16

block = Block(arg_types=(elt_type,elt_type,elt_type))
mulf = Mulf(
    block.args[0],
    block.args[1],
    FastMathFlagsAttr("fast"),
)
addf = Addf(
    block.args[2],
    mulf.results[0],
    FastMathFlagsAttr("fast"),
)
block.add_ops([
    mulf,
    addf,
    linalg.YieldOp(addf.results[0])
])

linalg_generic = linalg.Generic(
    (
        TestSSAValue(MemRefType(elt_type, [i, k])),
        TestSSAValue(MemRefType(elt_type, [k, j])),
    ),
    (TestSSAValue(MemRefType(elt_type, [i, j])),),
    Region(block),
    (
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(2)))
        ),
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(2),AffineExpr.dimension(1)))
        ),
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(1)))
        ),
    ),
    (
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.reduction(),
    ),
)

A_ty =MemRefType(elt_type, [i, k])
B_ty = MemRefType(elt_type, [k, j])
C_ty = MemRefType(elt_type, [i, j])
fun_block = Block(arg_types=[A_ty,B_ty,C_ty])
A = fun_block.args[0]
B = fun_block.args[1]
C = fun_block.args[2]

fZero = Constant(FloatAttr(0.0, f32))
linalg_fill = linalg.FillOp(inputs=(fZero.result,), outputs=(C,), res=[])

linalg_matmul = linalg.MatmulOp(
    inputs = (A,B),
    outputs = (C,),
)

fun_block.add_ops([
    fZero,
    linalg_fill,
    linalg_matmul,
    func.Return()
])

fun = func.FuncOp.from_region(
    "myfun",
    [A_ty,B_ty,C_ty],
    [],
    Region(fun_block)
)

fill_impl = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_fill,
    dims = {'i':i,'j':j},
    parallel_dims = ['i','j'],
    reduction_dims = [],
    vectors_size = vectors_size
)

fill_impl.tile("i",{'i1':4})
fill_impl.tile("j",{'j1':64})
fill_impl.interchange(['i','j','i1','j1'])
fill_impl.vectorize(['j1'])
fill_impl.parallelize(['i'])
fill_impl.unroll({'i1':4})

# e = fill_impl.compile(
#     print_source_ir=True,
#     print_transformed_ir=False,
#     print_lowered_ir = False,
#     print_assembly=False,
#     color = True,
#     debug = False,
# )

matmul_impl = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_matmul,
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
    vectors_size = vectors_size
)

matmul_impl.tile("i",{'i1':4})
matmul_impl.tile("j",{'j1':64})
matmul_impl.tile("k",{'k1':8})
matmul_impl.interchange(['i','j','k','k1','i1','j1'])
matmul_impl.vectorize(['j1'])
matmul_impl.parallelize(['i'])
matmul_impl.unroll({'i1':4,'k1':8})

impl_graph = MlirGraphImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    vectors_size = vectors_size,
    graph = fun,
    mlir_impls = [fill_impl,matmul_impl],
)

e = impl_graph.compile(
    print_source_ir=False,
    print_transformed_ir=True,
    print_lowered_ir = False,
    print_assembly=False,
    color = True,
    debug = False,
)
