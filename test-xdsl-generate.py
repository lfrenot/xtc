from setup_xdsl_mm import mm

impl = mm()

mlircode = impl.generate_without_compilation()
print(mlircode)
