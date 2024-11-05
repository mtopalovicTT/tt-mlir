// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include <vector>

namespace mlir::tt {

bool ReluIsLegal(const mlir::tt::LayoutAttr &inputLayout,
                 const mlir::tt::LayoutAttr &outputLayout);
size_t ReluGetOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                        const mlir::tt::LayoutAttr &outputLayout);

} // namespace mlir::tt