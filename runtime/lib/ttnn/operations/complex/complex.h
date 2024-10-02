// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_COMPLEX_H
#define TTNN_RUNTIME_COMPLEX_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::complex {
void run(const ::tt::target::ttnn::ComplexOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::complex

#endif
