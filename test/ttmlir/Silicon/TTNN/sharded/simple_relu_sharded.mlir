// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-to-ttnn-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<256x512xf32>) -> tensor<256x512xf32> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<256x512xf32>
    // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<256x512xf32>
  }
}
