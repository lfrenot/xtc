import utils
import numpy as np

__all__ = [
    "Operation",
    "Operators",
    "Operator",
    "OperatorMatmul",
]

tvm = utils.LazyImport("tvm")
te = utils.LazyImport("tvm.te")

class Operation():
    def __init__(self, operator, args):
        self.operator = operator
        self.args = args
        self.tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.params = None
        self.sch = None
        self.built = None

    def generate(self, schedule=""):
        self.params = self.operator.generate_op(*self.args)

    def schedule(self, schedule=""):
        self.sch = te.create_schedule(self.params[-1].op)
        if schedule:
            exec(schedule,
                 {"sch": self.sch, "obj": self.params},
                 {})

    def build(self):
        self.built = tvm.build(self.sch, self.params, self.tgt, name=self.operator.name)

    def load_module(self, dll):
        self.params = None
        self.sch = None
        self.built = tvm.runtime.load_module(dll)
        assert self.built.implements_function(self.operator.name), f"Loaded module does not correspond to operator"

    def run_eval(self, repeat=1, min_repeat_ms=0, number=1, validate=False):
        evaluator = self.built.time_evaluator(
            self.built.entry_name,
            self.dev,
            min_repeat_ms=min_repeat_ms,
            repeat=repeat,
            number=number,
        )
        inputs = [
            utils.np_init(shape=shape, dtype=dtype)
            for shape, dtype in zip(self.operator.inputs_dims(*self.args), self.operator.inputs_types(*self.args))]
        outputs = [
            np.empty(shape=shape, dtype=dtype)
            for shape, dtype in zip(self.operator.outputs_dims(*self.args), self.operator.outputs_types(*self.args))]
        tvm_inputs = [tvm.nd.array(t) for t in inputs]
        tvm_outputs = [tvm.nd.array(t) for t in outputs]
        if validate:
            ref_outs = [o.copy() for o in outputs]
            self.operator.reference_impl(*inputs, *ref_outs)
            self.built(*tvm_inputs, *tvm_outputs)
            for ref_out, out in zip(ref_outs, tvm_outputs):
                if not np.allclose(ref_out, out.numpy()):
                    return [], 1, "Error in validation: outputs differ"
        results = evaluator(*tvm_inputs, *tvm_outputs).results
        return results, 0, ""

    def run(self):
        results, code, error = self.run_eval(repeat=1, min_repeat_ms=0, number=1)
        return min(results) if code == 0 else error

    def lower(self):
        return tvm.lower(self.sch, self.params, simple_mode=True)


class Operator():
    name = "undef"
    @staticmethod
    def generate_op(i, j, k, dtype):
        raise Exception("unimplemneted")


class OperatorMatmul(Operator):
    name = "matmul"
    @staticmethod
    def generate_op(i, j, k, dtype):
        A = te.placeholder((i, k), name="A")
        B = te.placeholder((k, j), name="B")
        
        k = te.reduce_axis((0, k), "k")
        O = te.compute(
            (i, j),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            attrs={"layout_free_placeholders": [B]},
            name="O"
        )
        return A, B, O

    @staticmethod
    def inputs_dims(i, j, k, dtype):
        return (i, k), (k, j)

    @staticmethod
    def inputs_types(i, j, k, dtype):
        return dtype, dtype

    @staticmethod
    def outputs_dims(i, j, k, dtype):
        return (i, j),

    @staticmethod
    def outputs_types(i, j, k, dtype):
        return dtype,

    @staticmethod
    def reference_impl(*args):
        np.matmul(args[0], args[1], out=args[2])

class Operators():
    matmul = OperatorMatmul


def test_matmul(tdir="."):
    operation = Operation(Operators.matmul, (256, 256, 512, "float32"))
    operation.generate()
    operation.schedule()
    operation.build()
    operation.built.export_library(f"{tdir}/tvm-payload.so")
    time = operation.run()
    print(f"Execution time {operation.built.entry_name}: {time} secs")

def test_load_and_run(tdir="."):
    modlib = tvm.runtime.load_module(f"{tdir}/tvm-payload.so")
    operation = Operation(Operators.matmul, (256, 256, 512, "float32"))
    operation.built = modlib
    time = operation.run()
    print(f"Execution time {operation.built.entry_name}: {time} secs")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tdir:
        test_matmul(tdir)
        test_load_and_run(tdir)
