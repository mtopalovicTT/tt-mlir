// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_constant attributes {} {
  func.func public @test_bfloat16_scalar() -> tensor<bf16> {
    %0 = stablehlo.constant dense<3.0> : tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_bfloat16_empty() -> tensor<64x128xbf16> {
    %0 = stablehlo.constant dense<0.0> : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }

  func.func public @test_bfloat16_splat() -> tensor<64x128xbf16> {
    %0 = stablehlo.constant dense<3.0> : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
