from AbsImplementer import AbsImplementer

class PartialImplementer(AbsImplementer):

    def __init__(
            self,
            mlir_install_dir: str,
            source_path: str,
            payload_name: str
    ):
        super().__init__(mlir_install_dir,payload_name)
        self.source_path = source_path

    def glue(self):
        f = open(self.source_path, "r")
        source_ir = f.read()
        f.close()
        return source_ir

    def payload(self,m,elt_type):
        assert(False)

    def uniquely_match(self):
        assert(False)

    def materialize_schedule(self):
        assert(False)

    def main(self):
        assert(False)

    def build_rtclock(self):
        assert(False)

    def build_printF64(self):
        assert(False)
