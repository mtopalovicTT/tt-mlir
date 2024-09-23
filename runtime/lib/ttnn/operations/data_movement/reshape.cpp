// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {

static ::ttnn::Tensor tilize(::ttnn::Tensor const &input) {
  // NOLINTNEXTLINE
  return ::ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

static ::ttnn::Tensor untilize(::ttnn::Tensor const &input) {
  return ::ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

template <int32_t Rank>
static std::array<int32_t, Rank>
vectorToArray(const std::vector<int32_t> &vec) {
  if (vec.size() != Rank) {
    throw std::invalid_argument("Vector size does not match array size");
  }
  std::array<int32_t, Rank> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

template <int32_t Rank>
static ::ttnn::Tensor invoke_reshape(const ::ttnn::Tensor &tensor,
                                     const std::vector<int32_t> &shape) {
  if (tensor.get_layout() == ::ttnn::ROW_MAJOR_LAYOUT) {
    return ::ttnn::reshape(tensor, vectorToArray<Rank>(shape));
  }

  ::ttnn::Tensor in1 = untilize(tensor);
  ::ttnn::Tensor out = ::ttnn::reshape(in1, vectorToArray<Rank>(shape));
  return tilize(out);
}

void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  constexpr int32_t Rank1 = 1;
  constexpr int32_t Rank2 = 2;
  constexpr int32_t Rank3 = 3;
  constexpr int32_t Rank4 = 4;
  constexpr int32_t Rank5 = 5;

  ::ttnn::Tensor out;
  switch (fbShape->size()) {
  case Rank1:
    out = invoke_reshape<Rank1>(in, shape);
    break;
  case Rank2:
    out = invoke_reshape<Rank2>(in, shape);
    break;
  case Rank3:
    out = invoke_reshape<Rank3>(in, shape);
    break;
  case Rank4:
    out = invoke_reshape<Rank4>(in, shape);
    break;
  case Rank5:
    out = invoke_reshape<Rank5>(in, shape);
    break;
  default:
    throw std::invalid_argument("Unsupported rank for reshape");
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
