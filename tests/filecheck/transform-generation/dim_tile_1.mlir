// RUN: mlir-loop %s --no-alias --always-vectorize --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<512x128xf32>,
  %B: memref<128x1024xf32>,
  %C: memref<512x1024xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<512x1024xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"i" = ["i1"], "k" = ["k1"], "j" = ["j1"]},
      loop.tiles_sizes = {i1 = 64, j1=1, k1 = 4},
      loop.interchange = ["i","k", "j", "i1", "k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {"k1" = 4}
    }
    ins(%A, %B : memref<512x128xf32>, memref<128x1024xf32>)
    outs(%C : memref<512x1024xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<512x128xf32> {llvm.noalias}, %arg1: memref<128x1024xf32> {llvm.noalias}, %arg2: memref<512x1024xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg2 : memref<512x1024xf32>)
// CHECK-NEXT:      linalg.matmul {__id0__} ins(%arg0, %arg1 : memref<512x128xf32>, memref<128x1024xf32>) outs(%arg2 : memref<512x1024xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [64, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__id0__i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 0, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__id0__k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__id0__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__id0__i1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__id0__k1" : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      %3 = transform.structured.match attributes {__id0__k1} in %2 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %3 {factor = 4 : i64} : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
