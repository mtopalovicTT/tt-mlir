// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_maximum attributes {} {
  func.func public @test_maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
