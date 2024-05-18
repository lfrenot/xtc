from setup_mlir_mm import mm

impl = mm()

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
