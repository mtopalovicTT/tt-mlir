// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_constant attributes {} {
  func.func public @test_int32_scalar() -> tensor<i16> {
    %0 = stablehlo.constant dense<3> : tensor<i16>
    return %0 : tensor<i16>
  }

  func.func public @test_int32_empty() -> tensor<64x128xi16> {
    %0 = stablehlo.constant dense<0> : tensor<64x128xi16>
    return %0 : tensor<64x128xi16>
  }

  func.func public @test_int32_splat() -> tensor<64x128xi16> {
    %0 = stablehlo.constant dense<3> : tensor<64x128xi16>
    return %0 : tensor<64x128xi16>
  }
}
