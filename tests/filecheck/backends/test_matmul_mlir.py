# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_fill_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_reduce_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_fill_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "__xtc_id_C_fill_i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "__xtc_id_C_fill_j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_reduce_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "__xtc_id_C_reduce_k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "__xtc_id_C_reduce_i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "__xtc_id_C_reduce_j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "__xtc_id_C_reduce_i1" : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.vectorize_children_and_apply_patterns %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<1x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %cst_1 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c32 step %c1 {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          vector.transfer_write %cst_1, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {__xtc_id_C_fill_j}
# CHECK-NEXT:      } {__xtc_id_C_fill_i}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c4 step %c2 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview_3[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c16 {
# CHECK-NEXT:            %subview_6 = memref.subview %subview_2[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_5[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_8 = memref.subview %subview_4[%c0, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_7[%c0, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %0 = vector.transfer_read %subview_8[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:            %1 = vector.transfer_read %subview_6[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %2 = vector.transfer_read %subview_9[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %3 = vector.extract %1[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:            %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:            %5 = vector.broadcast %4 : f32 to vector<16xf32>
# CHECK-NEXT:            %6 = vector.extract %2[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:            %7 = vector.fma %5, %3, %6 : vector<16xf32>
# CHECK-NEXT:            %8 = vector.insert %7, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:            vector.transfer_write %8, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_4[%c1, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_7[%c1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %9 = vector.transfer_read %subview_10[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:            %10 = vector.transfer_read %subview_6[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %11 = vector.transfer_read %subview_11[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %12 = vector.extract %10[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:            %13 = vector.extract %9[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:            %14 = vector.broadcast %13 : f32 to vector<16xf32>
# CHECK-NEXT:            %15 = vector.extract %11[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:            %16 = vector.fma %14, %12, %15 : vector<16xf32>
# CHECK-NEXT:            %17 = vector.insert %16, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:            vector.transfer_write %17, %subview_11[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {__xtc_id_C_reduce_j}
# CHECK-NEXT:        } {__xtc_id_C_reduce_i}
# CHECK-NEXT:      } {__xtc_id_C_reduce_k}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'}
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
