import os,sys

sys.path.append('../')

from MlirImplementer import MlirImplementer

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import TensorType,MemRefType,f32
from xdsl.ir import Block

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 16

operands_types = [MemRefType(elt_type,shape)
                  for shape in [[i,k],[k,j],[i,j]]]
block0 = Block(arg_types=operands_types)
matmul = linalg.MemRefMatmulOp(
    inputs=(block0.args[0],block0.args[1]),
    outputs = (block0.args[2],),
)

def mm0():
    
    impl = MlirImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op = matmul,
        dims = {'i':i,'j':j,'k':k},
        parallel_dims = ['i','j'],
        reduction_dims = ['k'],
        vectors_size = vectors_size
    )
    
    return mm0

def mm1():

    impl = MlirImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op = matmul,
        dims = {'i':i,'j':j,'k':k},
        parallel_dims = ['i','j'],
        reduction_dims = ['k'],
        vectors_size = vectors_size
    )

    impl.tile("i",{'i1':8})
    impl.tile("j",{'j1':8})
    impl.tile("k",{'k1':8})
    impl.interchange(['i','j','k','i1','k1','j1'])
    impl.vectorize(['j1'])
    impl.parallelize(['i'])
    impl.unroll({'k1':8,'i1':8})

    return impl

def mm4():
    
    impl = MlirImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op = matmul,
        dims = {'i':i,'j':j,'k':k},
        parallel_dims = ['i','j'],
        reduction_dims = ['k'],
        vectors_size = vectors_size
    )

    impl.tile("i",{'i1':4})
    impl.tile("j",{'j1':64})
    impl.tile("k",{'k1':8})
    impl.interchange(['i','j','k','k1','i1','j1'])
    impl.vectorize(['j1'])
    impl.parallelize(['i'])
    impl.unroll({'i1':4,'k1':8})

    return impl
