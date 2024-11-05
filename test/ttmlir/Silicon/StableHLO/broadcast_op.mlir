// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_broadcast attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<64x128xf32> {mhlo.layout_mode = "default"}) -> (tensor<64x128xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1xf32>) -> tensor<64x128xf32>
    %1 = stablehlo.maximum %0, %arg1 : tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
