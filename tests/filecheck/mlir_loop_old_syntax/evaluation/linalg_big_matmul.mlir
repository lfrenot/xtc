// RUN: mlir-loop --old-syntax %s --no-alias --init-zero --vectors-size 4 --evaluate 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x2048xf32>,
  %B: memref<2048x8192xf32>,
  %C: memref<256x8192xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {"j" = {"j1" = 4096, "j2" = 32, "j3" = 16}, "i" = {"i1" = 8}, "k" = {"k1" = 1024}},
      loop.interchange = ["k", "j", "i", "j1", "k1", "i1", "j2", "j3"],
      loop.vectorize = ["j3"],
      loop.unroll = {j2 = 2, i1 = 8}
    }
    ins(%A, %B : memref<256x2048xf32>, memref<2048x8192xf32>)
    outs(%C : memref<256x8192xf32>)
  return
}

// CHECK: [[FLOAT:.*[0-9]+\.[0-9]+.*]]
