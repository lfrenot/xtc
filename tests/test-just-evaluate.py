import os
import sys

sys.path.append('../')

from PartialImplementer import PartialImplementer

source_path = "/tmp/test.mlir" if len(sys.argv) <= 1 else sys.argv[1]
payload_name = "payload0"
vectors_size = 16

home = os.environ.get("HOME","")

os.path.isfile(source_path)

impl = PartialImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_path = source_path,
    payload_name = payload_name,
    vectors_size=16,
)

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=True,
    color = True,
    debug = True,
)
print(e)
