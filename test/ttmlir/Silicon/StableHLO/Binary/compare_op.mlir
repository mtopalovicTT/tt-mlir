// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_compare attributes {} {
  func.func public @test_eq(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_ne(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  NE, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_ge(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  GE, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_gt(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_le(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  LE, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_lt(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
