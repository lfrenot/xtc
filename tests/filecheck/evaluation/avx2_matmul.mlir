// RUN: mlir-loop %s --evaluate --no-alias

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = {"i"=256,"j"=256},
        loop.parallel_dims = ["i","j"],
        loop.reduction_dims = [],
        loop.tiles_names = {"j" = ["j1"]},
        loop.tiles_sizes = {j1 = 8},
        loop.interchange = ["i","j","j1"],
        loop.vectorize = ["j1"],
        loop.parallelize = ["i"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = {"i"=256,"j"=256,"k"=512},
      loop.parallel_dims = ["i","j"],
      loop.reduction_dims = ["k"],
      loop.tiles_names = {"j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {k1 = 8}
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
