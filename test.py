from collections import OrderedDict

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import TensorType,f32

from xdsl.ir import Block

import Implementer
from Implementer import Implementer

i = 512
j = 128
k = 1024
elt_type = f32

operands_types = [TensorType(elt_type,shape) for shape in [[i,j],[j,k],[i,k]]]
block0 = Block(arg_types=operands_types)
matmul = linalg.MatmulOp((block0.args[0],block0.args[1]),(block0.args[2],),)
    
impl = Implementer(
    mlir_install_dir="/home/hpompougnac/bin/llvm-xdsl",
    source_op=matmul,
    dims = OrderedDict([('i',i),('k',k),('j',j)]),
    parallel_dims = ['i','k']
)
impl.tile("i",OrderedDict([('i1',8)]))
impl.tile("j",OrderedDict([('j1',4)]))
impl.tile("k",OrderedDict([('k1',4)]))
impl.interchange(['i','k','j','i1','k1','j1'])
impl.vectorize(['k1'])
impl.parallelize(['i'])
impl.unroll(OrderedDict([('i1',8)]))
stderr,stdout = impl.evaluate(
    print_source_ir=True,
    print_transformed_ir=False,
)
print(stderr)
print(stdout)

# print(impl.permutation)
# print(impl.tiles)
