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
    dims = OrderedDict([('i',i),('k',k),('j',j)])
)
impl.tile("i",OrderedDict([('i1',8)]))
impl.tile("j",OrderedDict([('j1',4)]))
impl.tile("k",OrderedDict([('k1',4)]))
# impl.interchange(['i','j1','k1','i1','j','k'])
# impl.vectorize(['k1'])
# impl.unroll(OrderedDict([('i1',64)]))
stderr,stdout = impl.evaluate(
    print_source_ir=True,
    print_transformed_ir=True
)
print(stderr)
print(stdout)
