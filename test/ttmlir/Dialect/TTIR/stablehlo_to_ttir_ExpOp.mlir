// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_exp attributes {} {
  func.func public @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.exponential %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.exp"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }
}