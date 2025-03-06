// RUN: mlir-loop %s --evaluate --no-alias

func.func @myfun(
  %I: memref<1x30x30x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x28x128xf32>
) {
  %cst = arith.constant 0.000 : f32
  linalg.fill
    {
      loop.dims = {"n"=1, "h"=28, "w"=28, "f"=128},
      loop.parallel_dims = ["n","h","w","f"],
      loop.reduction_dims = [],
      loop.tiles_names = {"f" = ["f1"]},
      loop.tiles_sizes = {f1 = 8},
      loop.interchange = ["n","h","w","f","f1"],
      loop.vectorize = ["f1"],
      loop.parallelize = ["h"]
    }
    ins(%cst : f32)
    outs(%O : memref<1x28x28x128xf32>)
  linalg.generic {
      indexing_maps = [
        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,
        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,
        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
          "reduction", "reduction", "reduction"],
      loop.dims = {"n"=1, "h"=28, "w"=28, "f"=128, "c"=64, "r"=3, "s"=3},
      loop.parallel_dims = ["x", "y", "f"],
      loop.reduction_dims = ["c", "w", "h"],
      loop.tiles_names = {"f" = ["f1"], "h" = ["h1", "h2"], "w" = ["w1"], "c" = ["c1"]},
      loop.tiles_sizes = {f1 = 64, h2 = 2, w1 = 2, c1 = 4, h1 = 14},
      loop.interchange = ["r", "s", "h", "w", "h1", "f", "c", "c1", "w1", "h2", "f1"],
      loop.vectorize = ["f1"],
      loop.unroll = {f1 = 4, h2 = 2, w1 = 2, c1 = 4}
    }
    ins(%I, %K : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>)
    outs(%O : memref<1x28x28x128xf32>)
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}
