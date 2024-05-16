import os

from MmMlirImplementer import MmMlirImplementer

source_path = "/tmp/test.mlir"

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024

impl = MmMlirImplementer(
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

mlircode = impl.generate_without_compilation()
f = open(source_path,'w')
f.write(mlircode)
f.close()

from PartialImplementer import PartialImplementer

impl = PartialImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_path = source_path,
    payload_name = impl.payload_name,
)

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color = True,
    debug = False,
)
print(e)
