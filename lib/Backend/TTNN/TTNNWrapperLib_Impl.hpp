// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNWrapper.hpp"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wctad-maybe-unsupported"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wvla-extension"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#pragma clang diagnostic ignored "-Wc++20-designator"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wstring-conversion"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wpessimizing-move"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#pragma clang diagnostic ignored "-Wc++23-extensions"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
#pragma clang diagnostic ignored "-Wundefined-inline"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wc++11-narrowing"

#define FMT_HEADER_ONLY
// #include "distributed/mesh_device.hpp"
// #include "host_api.hpp"
// #include "hostdevcommon/common_values.hpp"
// #include "ttnn/device.hpp"
// #include "ttnn/operations/ccl/all_gather/all_gather.hpp"
// #include "ttnn/operations/conv/conv2d/conv2d.hpp"
// #include "ttnn/operations/copy.hpp"
// #include "ttnn/operations/core/core.hpp"
// #include "ttnn/operations/creation.hpp"
// #include "ttnn/operations/data_movement/concat/concat.hpp"
// #include "ttnn/operations/data_movement/permute/permute.hpp"
// #include "ttnn/operations/eltwise/binary/binary.hpp"
// #include "ttnn/operations/eltwise/binary/binary_composite.hpp"
// #include "ttnn/operations/eltwise/unary/unary.hpp"
// #include "ttnn/operations/embedding/embedding.hpp"
// #include "ttnn/operations/matmul/matmul.hpp"
// #include "ttnn/operations/normalization/softmax/softmax.hpp"
// #include "ttnn/operations/pool/maxpool/max_pool2d.hpp"
// #include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#pragma clang diagnostic pop

namespace mlir::tt {

::ttnn::SimpleShape get_tensor_shape(const mlir::MemRefType &memref);

} // namespace mlir::tt