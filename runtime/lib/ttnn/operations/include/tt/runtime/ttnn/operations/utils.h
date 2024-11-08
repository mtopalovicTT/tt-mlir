// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_OPERATIONS_UTILS_H
#define TTNN_RUNTIME_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

bool isOnHost(const ::ttnn::Tensor &tensor);

bool isOnDevice(const ::ttnn::Tensor &tensor);

bool isTilized(const ::tt::target::TensorRef *tensorRef);

bool inSystemMemory(const ::tt::target::TensorRef *tensorRef);

void updateTensorPool(ProgramTensorPool &tensorPool,
                      const ::ttnn::Tensor &tensor, uint32_t outputGlobalId);

::tt::target::MemorySpace
getMemorySpace(const ::tt::target::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef);

::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::MemoryConfigDesc *memcfg,
                   const ::tt::target::TensorRef *tensorRef);

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::DistributionStrategy *strategy);

} // namespace tt::runtime::ttnn::operations::utils
#endif
