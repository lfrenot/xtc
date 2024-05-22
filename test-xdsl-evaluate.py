from setup_xdsl_mm import mm1

impl = mm1()

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
