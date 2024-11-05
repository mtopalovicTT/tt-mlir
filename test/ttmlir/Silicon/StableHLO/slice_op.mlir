// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_subtract attributes {} {
  func.func @slice_op(%arg0: tensor<32x64xf32>) -> tensor<8x8xf32> {
    %result = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 0, 16>,
      limit_indices = array<i64: 16, 32>,
      strides = array<i64: 2, 2>
    } : (tensor<32x64xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
