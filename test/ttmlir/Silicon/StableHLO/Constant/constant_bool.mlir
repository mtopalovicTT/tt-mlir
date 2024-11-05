// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_constant attributes {} {
  func.func public @test_boolean_scalar() -> tensor<i1> {
    %0 = stablehlo.constant dense<true> : tensor<i1>
    return %0 : tensor<i1>
  }

  func.func public @test_boolean_splat() -> tensor<64xi1> {
    %0 = stablehlo.constant dense<true> : tensor<64xi1>
    return %0 : tensor<64xi1>
  }
}
