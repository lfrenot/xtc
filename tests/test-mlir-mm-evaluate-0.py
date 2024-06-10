from setup_mlir_mm import mm0

impl = mm0()

e = impl.evaluate(
    print_source_ir=True,
    print_transformed_ir=False,
    print_assembly=False,
    color = True,
    debug = False,
)

print(e)
