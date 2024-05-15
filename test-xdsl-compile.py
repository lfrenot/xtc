import os

from XdslImplementer import XdslImplementer

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024

impl = XdslImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    named_op_name="matmul",
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
    element_type = 'f32',
)

impl.tile("i",{'i1':8})
impl.tile("j",{'j1':8})
impl.tile("k",{'k1':8})
impl.interchange(['i','k','j','i1','k1','j1'])
impl.vectorize(['j1'])
impl.parallelize(['i'])
impl.unroll({'k1':8,'i1':8})

impl.compile(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color = True,
    debug = False,
)

