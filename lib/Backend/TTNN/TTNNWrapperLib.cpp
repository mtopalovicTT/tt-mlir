// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>

#include "TTNNWrapper.hpp"
#include "TTNNWrapperLib_Impl.hpp"

namespace mlir::tt {
void calculus(mlir::tt::LayoutAttr layout) {
  throw std::runtime_error("Not implemented");
}

void print_tensor_shape(const mlir::MemRefType &memref) {
  const auto shape = get_tensor_shape(memref);
  std::cout << shape << std::endl;
}

::ttnn::SimpleShape get_tensor_shape(const mlir::MemRefType &memref) {
  std::vector<uint32_t> shape;
  for (auto i = 0; i < memref.getRank(); i++) {
    shape.push_back(memref.getShape()[i]);
  }
  return ::ttnn::SimpleShape(shape);
}
} // namespace mlir::tt