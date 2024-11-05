// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_multiply attributes {} {
  func.func public @test_multiply(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
