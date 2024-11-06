// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::backend::ttnn {

struct ReluOpInterface {
  static bool IsLegal(const mlir::tt::LayoutAttr &inputLayout,
                      const mlir::tt::LayoutAttr &outputLayout);
  static size_t GetOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                             const mlir::tt::LayoutAttr &outputLayout);
};

} // namespace mlir::tt::backend::ttnn