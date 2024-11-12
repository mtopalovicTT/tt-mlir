// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsBackendInterfaces.cpp.inc"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

// // Relu backend interface
size_t
ReluOp::getOpPerfCycles(const std::vector<TensorConfigAttr> &input_layouts,
                        const TensorConfigAttr &output_layout) {
  // Implement a custom estimate for relu op cycles.
  return 5;
}

size_t ReluOp::getOpL1Usage(const std::vector<TensorConfigAttr> &input_layouts,
                            const TensorConfigAttr &output_layout) {
  // Implement a custom estimate for relu op L1 usage.
  return 10;
}

bool ReluOp::isOpLegal(const std::vector<TensorConfigAttr> &input_layouts,
                       const TensorConfigAttr &output_layout) {
  // Implement a custom check for relu op legality.
  return true;
}

} // namespace mlir::tt::ttnn
