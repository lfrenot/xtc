// RUN: mlir-loop %s --evaluate --no-alias

func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    {
      loop.dims = ["i","j"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"]},
      loop.tiles_sizes = {i1 = 4, j1 = 64},
      loop.interchange = ["i","j","i1","j1"],
      loop.vectorize = ["j1"],
      loop.parallelize = ["i"],
      loop.unroll = {i1 = 4}
    }
    ins(%cst : f32)
    outs(%C : memref<512x128xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 4, j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","i1","j1"],
      loop.vectorize = ["j1"],
      loop.parallelize = ["i"],
      loop.unroll = {i1 = 4, k1 = 8},
      loop.add_attributes = ["JoeDassin"]
    }
    ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>)
    outs(%C : memref<512x128xf32>)
  return
}
