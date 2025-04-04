// RUN: mlir-loop %s --no-alias --print-source-ir 2>&1 | filecheck %s

// Matmul I=78, J=16, K=32
// Schedule ttile: [ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10,12]); Seq(I) ]

// Fully hierarchical (without references)

func.func @myfun1(
  %A: memref<78x32xf32>,
  %B: memref<32x16xf32>,
  %C: memref<78x16xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<78x16xf32>)
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I[:30]" = {
          "J",
          "K",
          "K#1",
          "I#3" = {"Unroll"},
          "J#16" = {"Vectorize"}
        },
        "I[30:]" = {
          "J",
          "K",
          "K#1",
          "I#4" = {"Unroll"},
          "J#16" = {"Vectorize"}
        }
      }
    }
    ins(%A, %B : memref<78x32xf32>, memref<32x16xf32>) outs(%C : memref<78x16xf32>)
  return
}
