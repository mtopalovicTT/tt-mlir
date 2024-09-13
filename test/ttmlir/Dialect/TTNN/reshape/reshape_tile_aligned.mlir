// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x32x2x32xbf16> {
    %0 = tensor.empty() : tensor<4x32x2x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [4: i32, 32: i32, 2: i32, 32: i32] , operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x2x32x32xbf16>, tensor<4x32x2x32xbf16>) -> tensor<4x32x2x32xbf16>
    return %1 : tensor<4x32x2x32xbf16>
  }
}
