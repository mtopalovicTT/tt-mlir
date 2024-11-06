// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Backend/TTNN/TTNNWrapper.hpp"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsBackendInterfaces.cpp.inc"
#include <cassert>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

// // Relu backend interface
size_t ReluOp::getOpPerfCycles(const std::vector<tt::LayoutAttr> &input_layouts,
                               const tt::LayoutAttr &output_layout) {
  // Implement a custom estimate for relu op cycles.
  return 5;
}

size_t ReluOp::getOpL1Usage(const std::vector<tt::LayoutAttr> &input_layouts,
                            const tt::LayoutAttr &output_layout) {

  assert(input_layouts.size() == 1);
  return backend::ttnn::ReluOpInterface::GetOpL1Usage(input_layouts[0],
                                                      output_layout);
}

bool ReluOp::isOpLegal(const std::vector<tt::LayoutAttr> &input_layouts,
                       const tt::LayoutAttr &output_layout) {
  assert(input_layouts.size() == 1);
  return backend::ttnn::ReluOpInterface::IsLegal(input_layouts[0],
                                                 output_layout);
}

} // namespace mlir::tt::ttnn
