from setup_mlir_mm import mm4

impl = mm4()

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
