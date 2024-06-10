from setup_mlir_mm import mm1

source_path = "/tmp/test.mlir"
vectors_size = 16

impl = mm1()

mlircode = impl.generate_without_compilation()
f = open(source_path,'w')
f.write(mlircode)
f.close()

from PartialImplementer import PartialImplementer
import os

home = os.environ.get("HOME","")

impl = PartialImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_path = source_path,
    payload_name = impl.payload_name,
    vectors_size = vectors_size,
)

impl.compile(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=False,
    color = True,
    debug = False,
)

