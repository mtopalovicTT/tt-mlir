// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_convert attributes {} {
  func.func public @test_convert(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}

module @jit_eltwise_add attributes {} {
  func.func public @test_add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    %1 = stablehlo.convert %arg1 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    %2 = stablehlo.add %0, %1 : tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
}
