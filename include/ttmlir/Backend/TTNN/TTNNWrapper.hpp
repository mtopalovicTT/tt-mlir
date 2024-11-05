// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include <vector>

namespace mlir::tt {
void calculus(mlir::tt::LayoutAttr layout);

void print_tensor_shape(const mlir::MemRefType &memref);

} // namespace mlir::tt