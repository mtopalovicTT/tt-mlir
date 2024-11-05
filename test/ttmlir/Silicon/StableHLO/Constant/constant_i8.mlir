// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_constant attributes {} {
  func.func public @test_int32_scalar() -> tensor<i8> {
    %0 = stablehlo.constant dense<3> : tensor<i8>
    return %0 : tensor<i8>
  }

  func.func public @test_int32_empty() -> tensor<64x128xi8> {
    %0 = stablehlo.constant dense<0> : tensor<64x128xi8>
    return %0 : tensor<64x128xi8>
  }

  func.func public @test_int32_splat() -> tensor<64x128xi8> {
    %0 = stablehlo.constant dense<3> : tensor<64x128xi8>
    return %0 : tensor<64x128xi8>
  }
}
