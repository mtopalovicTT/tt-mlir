// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_get_dimension_size attributes {} {
  func.func public @test_get_dimension_size(%arg0: tensor<64x128xf32>) -> tensor<i32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<64x128xf32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
