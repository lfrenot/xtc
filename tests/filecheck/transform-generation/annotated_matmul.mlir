// RUN: mlir-loop %s --no-alias --always-vectorize --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<512x128xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.add_attributes = ["JoeDassin"]
    }
    ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>)
    outs(%C : memref<512x128xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<512x1024xf32> {llvm.noalias}, %arg1: memref<1024x128xf32> {llvm.noalias}, %arg2: memref<512x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<1x1xf32>
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %c1024 = arith.constant 1024 : index
// CHECK-NEXT:      %c128 = arith.constant 128 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c512 = arith.constant 512 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst_1 = arith.constant dense<0.000000e+00> : vector<512x128xf32>
// CHECK-NEXT:      vector.transfer_write %cst_1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<512x128xf32>, memref<512x128xf32>
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [1, 1024] [1, 1] : memref<512x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
// CHECK-NEXT:        %subview_2 = memref.subview %arg1[0, 0] [1024, 128] [1, 1] : memref<1024x128xf32> to memref<1024x128xf32, strided<[128, 1]>>
// CHECK-NEXT:        %subview_3 = memref.subview %arg2[%arg3, 0] [1, 128] [1, 1] : memref<512x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c128 step %c1 {
// CHECK-NEXT:          %subview_4 = memref.subview %subview_2[0, %arg4] [1024, 1] [1, 1] : memref<1024x128xf32, strided<[128, 1]>> to memref<1024x1xf32, strided<[128, 1], offset: ?>>
// CHECK-NEXT:          %subview_5 = memref.subview %subview_3[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c1024 step %c1 {
// CHECK-NEXT:            %subview_6 = memref.subview %subview[0, %arg5] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
// CHECK-NEXT:            %subview_7 = memref.subview %subview_4[%arg5, 0] [1, 1] [1, 1] : memref<1024x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
// CHECK-NEXT:            %0 = vector.transfer_read %subview_6[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[1024, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %1 = vector.transfer_read %subview_7[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[128, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %2 = vector.transfer_read %subview_5[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[128, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %3 = vector.extract %1[0] : vector<1xf32> from vector<1x1xf32>
// CHECK-NEXT:            %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %5 = vector.broadcast %4 : f32 to vector<1xf32>
// CHECK-NEXT:            %6 = vector.extract %2[0] : vector<1xf32> from vector<1x1xf32>
// CHECK-NEXT:            %7 = vector.fma %5, %3, %6 : vector<1xf32>
// CHECK-NEXT:            %8 = vector.insert %7, %cst [0] : vector<1xf32> into vector<1x1xf32>
// CHECK-NEXT:            vector.transfer_write %8, %subview_5[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, memref<1x1xf32, strided<[128, 1], offset: ?>>
// CHECK-NEXT:          } {__id0__k}
// CHECK-NEXT:        } {__id0__j}
// CHECK-NEXT:      } {JoeDassin, __id0__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
