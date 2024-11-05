// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" | \
// RUN:     ttmlir-translate --ttnn-to-flatbuffer > %t.ttnn

module @jit_eltwise_compare attributes {} {
  func.func public @logical_and(%arg0: tensor<64x128xi1>, %arg1: tensor<64x128xi1>) -> tensor<64x128xi1> {
    %0 = stablehlo.and  %arg0, %arg1 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @logical_or(%arg0: tensor<64x128xi1>, %arg1: tensor<64x128xi1>) -> tensor<64x128xi1> {
    %0 = stablehlo.or  %arg0, %arg1 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

func.func public @logical_not(%arg0: tensor<64x128xi1>) -> tensor<64x128xi1> {
    %0 = stablehlo.not  %arg0 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

func.func public @logical_not_scalar(%arg0: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.not  %arg0 : tensor<i1>
    return %0 : tensor<i1>
  }
}
