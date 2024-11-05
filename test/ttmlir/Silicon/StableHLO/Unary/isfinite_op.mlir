// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_isfinite attributes {} {
  func.func public @test_isfinite(%arg0: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.is_finite %arg0 : (tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
