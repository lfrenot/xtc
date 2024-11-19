// RUN: mlir-loop %s --print-source-ir --no-alias 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %Bquant: memref<512x256xi8>,
  %C: memref<256x256xf32>
) {

  // Here, we initialize the result memref
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
    
  // The following describes a dequantization operation
  %B = memref.alloc(): memref<512x256xf32>
  linalg.generic {
    loop.dims = {"k"=512,"j"=256},
    loop.parallel_dims = ["k","j"],
    loop.tiles_names = {"j" = ["j1"], "k" = ["k1"]},
    loop.tiles_sizes = {j1 = 64, k1 = 8},
    loop.interchange = ["k","j","k1","j1"],
    loop.vectorize = ["j1"],
    loop.unroll = {k1 = 8},
    indexing_maps = [
      affine_map<(d0,d1) -> (d0,d1)>,
      affine_map<(d0,d1) -> (d0,d1)>
    ],
    iterator_types = ["parallel","parallel"]
  } ins(%Bquant : memref<512x256xi8>) outs(%B : memref<512x256xf32>) {
  ^0(%2 : i8, %3 : f32):
    %4 = arith.constant 1.000000e-01 : f32
    %5 = arith.constant 1 : i32
    %6 = arith.sitofp %2 : i8 to f32
    %7 = arith.sitofp %5 : i32 to f32
    %8 = arith.subf %6, %7 : f32
    %9 = arith.mulf %8, %4 : f32
    linalg.yield %9 : f32
  }

  // The matmul itself
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
    
  memref.dealloc %B: memref<512x256xf32>
  return
}

// CHECK: TODO
