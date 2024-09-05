// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#l1_block_sharded = #tt.operand_constraint<l1_block_sharded>

func.func @subtract(%arg0: tensor<1792x256xf32>, %arg1: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.subtract"[[C:.*]]
  %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @div(%arg0: tensor<1792x256xf32>, %arg1: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.div"[[C:.*]]
  %1 = "ttir.div"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @multiply(%arg0: tensor<1792x256xf32>, %arg1: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
  %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @relu(%arg0: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @ge(%arg0: tensor<1792x256xf32>, %arg1: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.ge"[[C:.*]]
  %1 = "ttir.ge"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @reshape(%arg0: tensor<4x2x1792x256xbf16>) -> tensor<2x4x1792x256xbf16> {
  %0 = tensor.empty() : tensor<2x4x1792x256xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 1792: i32, 256: i32] , operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<4x2x1792x256xbf16>, tensor<2x4x1792x256xbf16>) -> tensor<2x4x1792x256xbf16>
  return %1 : tensor<2x4x1792x256xbf16>
}

func.func @squeeze(%arg0: tensor<1x2x1x1792x256xbf16>) -> tensor<1x2x1792x256xbf16> {
  %0 = tensor.empty() : tensor<1x2x1792x256xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.squeeze"(%arg0, %0) <{dim = 2 : si32, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1x2x1x1792x256xbf16>, tensor<1x2x1792x256xbf16>) -> tensor<1x2x1792x256xbf16>
  return %1 : tensor<1x2x1792x256xbf16>
}

func.func @reciprocal(%arg0: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.reciprocal"[[C:.*]]
  %1 = "ttir.reciprocal"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @sigmoid(%arg0: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.sigmoid"[[C:.*]]
  %1 = "ttir.sigmoid"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @sqrt(%arg0: tensor<1792x256xf32>) -> tensor<1792x256xf32> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xf32>
  // CHECK: %[[C:.*]] = "ttnn.sqrt"[[C:.*]]
  %1 = "ttir.sqrt"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xf32>, tensor<1792x256xf32>) -> tensor<1792x256xf32>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %1 : tensor<1792x256xf32>
}

func.func @softmax(%arg0: tensor<1792x256xbf16>) -> tensor<1792x256xbf16> {
  // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1792x256xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for positive dimension attribute
  %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xbf16>, tensor<1792x256xbf16>) -> tensor<1792x256xbf16>
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %2 = tensor.empty() : tensor<1792x256xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for negative dimension attribute
  %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<1792x256xbf16>, tensor<1792x256xbf16>) -> tensor<1792x256xbf16>
  // CHECK: "ttnn.close_device"[[C:.*]]
  return %3 : tensor<1792x256xbf16>
}

/////////////////////////////////////////
// Unsupported eltwise ops with sharding
//  * Concat: Sharded concat requires ROW MAJOR layout
/////////////////////////////////////////
