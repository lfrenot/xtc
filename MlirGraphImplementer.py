from xdsl.ir import Operation
from mlir.dialects import (
    builtin,
    func,
)
from mlir.ir import (
    InsertionPoint,
)

from AbsImplementer import AbsImplementer
from MlirImplementer import MlirImplementer
import mlir_packing

class MlirGraphImplementer(AbsImplementer):

    def __init__(
            self,
            mlir_install_dir: str,
            vectors_size: int,
            graph: Operation,
            mlir_impls: list[MlirImplementer]
    ):
        super().__init__(mlir_install_dir,vectors_size)
        self.mlir_impls = mlir_impls
        mlir_graph = func.FuncOp.parse(str(graph),context = self.ctx)
        ip = InsertionPoint.at_block_begin(self.module.body)
        ip.insert(mlir_graph)

    def np_inputs_spec(self):
        pass

    def np_outputs_spec(self):
        pass

    def reference_impl(self, *operands):
        pass

    def materialize_schedule(self,input_var):

        tiling_block = []
        last_tiled_loop = None
        for impl in self.mlir_impls:
            tiling_instrs, last_tiled_loop = impl.materialize_tiling(input_var)
            tiling_block += tiling_instrs

        vect_instrs,vectorized = self.mlir_impls[0].normalize_and_vectorize(last_tiled_loop)

        unroll_block = []
        to_unroll = vectorized
        for impl in self.mlir_impls:
            unroll_instrs,to_unroll = impl.materialize_unrolling(to_unroll)
            unroll_block += unroll_instrs
        
        return tiling_block + vect_instrs + unroll_block
    
    def integrate(self):
        assert(not self.integrated)
        mlir_packing.integrate([self],self.module,self.ctx)
        self.integrated = True
