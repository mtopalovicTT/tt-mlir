// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_constant attributes {} {
  func.func public @test_float_scalar() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_float_empty() -> tensor<64xf32> {
    %0 = stablehlo.constant dense<0.0> : tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  func.func public @test_float_splat() -> tensor<64xf32> {
    %0 = stablehlo.constant dense<0.3> : tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
