// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_dot_general attributes {} {
  func.func public @test_dot_general(%arg0 : tensor<16x32xf32>, %arg1 : tensor<32x8xf32>) -> tensor<16x8xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}
