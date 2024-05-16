import os

from MmMlirSepInitImplementer import MmMlirSepInitImplementer

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024

impl = MmMlirSepInitImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k']
)

impl.tile("i",{'i1':8})
impl.tile("j",{'j1':8})
impl.tile("k",{'k1':8})
impl.interchange(['i','j','k','i1','k1','j1'])
impl.vectorize(['j1'])
impl.parallelize(['i'])
impl.unroll({'k1':8,'i1':8})

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=False,
    color = True,
    debug = False,
)
print(e)
