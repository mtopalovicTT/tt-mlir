// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

//===----------------------------------------------------------------------===//
// IndexOp decomposition
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_index_ttir
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
struct IndexToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexOp> {
  using OpConversionPattern<ttir::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasRank())
      return failure();

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == op.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(adaptor.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(adaptor.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(adaptor.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    auto newOp = rewriter.create<ttir::SliceOp>(
        op.getLoc(), op.getType(), adaptor.getInput(), adaptor.getOutput(),
        rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
        rewriter.getArrayAttr(steps), adaptor.getOperandConstraints());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
// ANCHOR_END: adding_an_op_index_ttir

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

using TransposeDims = std::tuple<int64_t, int64_t>;

template <uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template <uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(DenseIntElementsAttr paddingAttr) {
  PaddingMatrix<NDims> paddingMatrix;
  std::vector<int64_t> paddingFlattened(paddingAttr.value_begin<int64_t>(),
                                        paddingAttr.value_end<int64_t>());

  for (uint32_t i = 0; i < 2 * NDims; i += 2) {
    paddingMatrix[i / 2] = {paddingFlattened[i], paddingFlattened[i + 1]};
  }
  return paddingMatrix;
}
/*
 * The following functions are used to generate the transpose operations needed
 * to convert a convolution operation to the specific op definitions for a
 * ConvNdOp for any N spatial dimensions.
 *
 * All convolutions will have a batch and feature dimension, and the kernel will
 * have an input and output feature dimension. The spatial dimensions can be
 * represented by non-negative integers.
 */
enum ConvolutionDimension { BATCH = -1, FEATURE = -2, INVALID_DIM = -3 };

enum ConvolutionKernelDimension {
  INPUT_FEATURES = -1,
  OUTPUT_FEATURES = -2,
  INVALID_KERNEL_DIM = -3
};

/*
 * Generates a sequence of dims in which to transpose to make current_layout
 * match desired_layout
 *
 * Ex: if current_layout = [0, 1, 2, 3] and desired_layout = [0, 2, 3, 1]
 * then the function will return [(1, 2), (2, 3)] because when we swap
 * current_layout[1] with current_layout[2] we get [0, 2, 1, 3], and then when
 * we swap current_layout[2] with current_layout[3] we get [0, 2, 3, 1], which
 * is the desired layout
 */
static std::vector<TransposeDims>
generateTransposeIndices(std::vector<int64_t> current_layout,
                         const std::vector<int64_t> desired_layout) {
  std::vector<TransposeDims> transpose_indices;
  for (int64_t i = 0; i < static_cast<int64_t>(current_layout.size()); i++) {
    if (current_layout[i] != desired_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(current_layout.begin(), current_layout.end(),
                               desired_layout[i]) -
                     current_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(current_layout[dim0], current_layout[dim1]);
    }
  }

  return transpose_indices;
}

/*
 * This function will use a sequence of transpose indices to
 * generate the actual transpose operations descrbibed by them.
 *
 * It takes an input to apply these transposes to and returns the
 * result at the end of the sequence
 */
static Value
generateTransposeSequence(Value input, PatternRewriter &rewriter,
                          std::vector<TransposeDims> transpose_indices,
                          ::mlir::ArrayAttr operand_constraints) {
  for (auto [dim0, dim1] : transpose_indices) {

    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto output_shape = input_type.getShape().vec();
    std::swap(output_shape[dim0], output_shape[dim1]);

    auto dim0_attr = rewriter.getSI32IntegerAttr(dim0);
    auto dim1_attr = rewriter.getSI32IntegerAttr(dim1);

    auto output_type = RankedTensorType::get(
        output_shape, input_type.getElementType(), input_type.getEncoding());

    auto dps_output = rewriter.create<tensor::EmptyOp>(
        input.getLoc(), output_shape, output_type.getElementType());
    input = rewriter
                .create<ttir::TransposeOp>(input.getLoc(), output_type, input,
                                           dps_output, dim0_attr, dim1_attr,
                                           operand_constraints)
                .getResult();
  }

  return input;
}

/*
 * This function will generate the transpose indices needed to convert a
 * convolution input to a desired layout. The reason for the separate
 * function is to encapsulate the logic for constructuring the input_layout
 */
static std::vector<TransposeDims> generateConvTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_layout) {

  std::vector<int64_t> input_layout(ttnn_convolution_layout.size(),
                                    ConvolutionDimension::INVALID_DIM);
  input_layout[op.getConvolutionLayout().getInputBatchDimension()] =
      ConvolutionDimension::BATCH;
  input_layout[op.getConvolutionLayout().getInputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getInputSpatialDimensions()) {
    input_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  return generateTransposeIndices(input_layout, ttnn_convolution_layout);
}

/*
 * This function will generate the transpose indices needed to convert a
 * convolution input to a desired layout. The reason for the separate
 * function is to encapsulate the logic for constructuring the kernel_layout
 */
static std::vector<TransposeDims> generateConvKernelTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_kernel_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> kernel_layout(
      ttnn_convolution_kernel_layout.size(),
      ConvolutionKernelDimension::INVALID_KERNEL_DIM);
  kernel_layout[op.getConvolutionLayout().getKernelOutputFeatureDimension()] =
      ConvolutionKernelDimension::OUTPUT_FEATURES;
  kernel_layout[op.getConvolutionLayout().getKernelInputFeatureDimension()] =
      ConvolutionKernelDimension::INPUT_FEATURES;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getKernelSpatialDimensions()) {
    kernel_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  return generateTransposeIndices(kernel_layout,
                                  ttnn_convolution_kernel_layout);
}

struct ConvolutionToConv2dPattern
    : public OpConversionPattern<ttir::ConvolutionOp> {
public:
  using OpConversionPattern<ttir::ConvolutionOp>::OpConversionPattern;

  constexpr static uint32_t numSpatialDims = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  const std::vector<int64_t> conv2d_layout = {
      ConvolutionDimension::BATCH, SPATIAL_DIM_HEIGHT, SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE};
  // OIHW
  const std::vector<int64_t> conv2d_kernel_layout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES, SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH};
  LogicalResult isConv2d(ttir::ConvolutionOp op) const {

    // Conv2d will have 2 spatial dimensions

    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getOutputSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getKernelSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");

    if (op.getConvolutionLayout().getInputSpatialDimensions().size() !=
        numSpatialDims) {
      return failure();
    }

    // Not currently supporting window reversal
    std::vector<bool> window_reversal(op.getWindowReversal().begin(),
                                      op.getWindowReversal().end());
    for (bool reversed : window_reversal) {
      if (reversed) {
        return failure();
      }
    }

    // Not currently support batch groups
    if (op.getBatchGroupCount() != 1) {
      return failure();
    }

    return success();
  }

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(isConv2d(op))) {
      return failure();
    }

    auto stride_height_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_HEIGHT]);
    auto stride_width_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_WIDTH]);
    auto dilation_height_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT]);
    auto dilation_width_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH]);

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto padding_matrix =
        getPaddingMatrix<numSpatialDims>(adaptor.getPadding());
    auto padding_top_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][0]);
    auto padding_bottom_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][1]);
    auto padding_left_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][0]);
    auto padding_right_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][1]);

    auto groups_attr =
        rewriter.getSI32IntegerAttr(adaptor.getFeatureGroupCount());

    auto output_shape = op.getResult().getType().getShape().vec();
    std::vector<int64_t> new_output_shape = {
        output_shape[adaptor.getConvolutionLayout().getOutputBatchDimension()],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputFeatureDimension()]};

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType =
        inputType.cloneWith(new_output_shape, inputType.getElementType());

    auto convDPSOutput = rewriter.create<tensor::EmptyOp>(
        adaptor.getInput().getLoc(), new_output_shape,
        outputType.getElementType());

    auto transpose_indices = generateConvTransposeIndices(op, conv2d_layout);
    Value input = generateTransposeSequence(adaptor.getInput(), rewriter,
                                            transpose_indices,
                                            adaptor.getOperandConstraints());

    auto kernel_transpose_indices =
        generateConvKernelTransposeIndices(op, conv2d_kernel_layout);
    Value weight = generateTransposeSequence(adaptor.getWeight(), rewriter,
                                             kernel_transpose_indices,
                                             adaptor.getOperandConstraints());
    ttir::Conv2dOp new_conv = rewriter.create<ttir::Conv2dOp>(
        op.getLoc(), outputType, input, weight, adaptor.getBias(),
        convDPSOutput, stride_height_attr, stride_width_attr,
        dilation_height_attr, dilation_width_attr, groups_attr,
        padding_left_attr, padding_right_attr, padding_top_attr,
        padding_bottom_attr, adaptor.getOperandConstraints());

    // Applying the transposes in reverse order to the output will restore the
    // tensor to the original layout
    std::reverse(transpose_indices.begin(), transpose_indices.end());
    Value output = generateTransposeSequence(new_conv.getResult(), rewriter,
                                             transpose_indices,
                                             adaptor.getOperandConstraints());

    rewriter.replaceOp(op, output);

    return success();
  }
};

struct PoolingToPool2dPattern : public OpConversionPattern<ttir::PoolingOp> {
public:
  using OpConversionPattern<ttir::PoolingOp>::OpConversionPattern;

  std::vector<int64_t> getIndicesOfSpatialDims(ttir::PoolingOp op) const {
    std::vector<int64_t> spatial_dims;
    for (int64_t i = 0;
         i < static_cast<int64_t>(op.getWindowDimensions().size()); i++) {
      if (op.getWindowDimensions()[i] > 1) {
        spatial_dims.push_back(i);
      }
    }
    return spatial_dims;
  }

  LogicalResult canDecompose2DPoolingOp(ttir::PoolingOp op) const {

    // Window dimensions must be 4 in length
    if (op.getWindowDimensions().size() != 4) {
      return failure();
    }

    // Window strides must be 4 in length
    if (op.getWindowStrides().size() != 4) {
      return failure();
    }

    // Operand rank(s) must be 4
    for (Value operand : op.getInputs()) {
      auto operand_type = mlir::cast<mlir::RankedTensorType>(operand.getType());
      if (operand_type.getRank() != 4) {
        return failure();
      }
    }

    // Exactly two of the window dimensions must be greater than 1
    std::vector<int64_t> true_window_dimensions_indices =
        getIndicesOfSpatialDims(op);

    if (true_window_dimensions_indices.size() != 2) {
      return failure();
    }

    // Exactly two of the window strides must be greater than 1
    std::vector<int64_t> true_window_stride_indices;
    for (int64_t i = 0; i < static_cast<int64_t>(op.getWindowStrides().size());
         i++) {
      if (op.getWindowStrides()[i] > 1) {
        true_window_stride_indices.push_back(i);
      }
    }

    if (true_window_stride_indices.size() != 2) {
      return failure();
    }

    // The indices of the true window dimensions and strides must be the same
    if ((true_window_dimensions_indices[0] != true_window_stride_indices[0] ||
         true_window_dimensions_indices[1] != true_window_stride_indices[1]) &&
        (true_window_dimensions_indices[0] != true_window_stride_indices[1] ||
         true_window_dimensions_indices[1] != true_window_stride_indices[0])) {
      return failure();
    }

    // Padding must be 8 in length
    if (op.getPadding().size() != 8) {
      return failure();
    }

    return success();
  }

  template <typename PoolOpType>
  void rewritePool2d(ttir::PoolingOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    const int64_t SPATIAL_H = -3;
    const int64_t SPATIAL_W = -2;
    const int64_t NON_SPATIAL = -1;

    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getInputs()[0].getType());
    assert(inputType.getRank() == 4 && "Input must be 4D tensor");
    std::vector<int64_t> desired_layout(inputType.getRank(), NON_SPATIAL);
    desired_layout[inputType.getRank() - 3] = SPATIAL_H;
    desired_layout[inputType.getRank() - 2] = SPATIAL_W;

    int64_t non_spatial_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(desired_layout.size()); i++) {
      if (desired_layout[i] == NON_SPATIAL) {
        desired_layout[i] = non_spatial_count;
        non_spatial_count++;
      }
    }

    std::vector<int64_t> spatial_dims = getIndicesOfSpatialDims(op);

    std::vector<int64_t> current_layout(inputType.getRank(), NON_SPATIAL);
    current_layout[spatial_dims[0]] = SPATIAL_H;
    current_layout[spatial_dims[1]] = SPATIAL_W;

    non_spatial_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(current_layout.size()); i++) {
      if (current_layout[i] == NON_SPATIAL) {
        current_layout[i] = non_spatial_count;
        non_spatial_count++;
      }
    }

    auto transpose_indices =
        generateTransposeIndices(current_layout, desired_layout);

    auto kernel_height_attr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatial_dims[0]]));
    auto kernel_width_attr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatial_dims[1]]));

    auto stride_height_attr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatial_dims[0]]));

    auto stride_width_attr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatial_dims[1]]));

    auto dilation_height_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatial_dims[0]]);
    auto dilation_width_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatial_dims[1]]);
    auto ceil_mode_attr = rewriter.getBoolAttr(false);

    auto padding_top_attr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatial_dims[0]]);
    auto padding_bottom_attr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatial_dims[0] + 1]);
    auto padding_left_attr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatial_dims[1]]);
    auto padding_right_attr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatial_dims[1] + 1]);
    auto operand_constraints = adaptor.getOperandConstraints();

    std::vector<Value> outputs;
    for (Value input : adaptor.getInputs()) {
      input = generateTransposeSequence(input, rewriter, transpose_indices,
                                        operand_constraints);

      auto outputType = mlir::cast<RankedTensorType>(op.getResult(0).getType());
      auto newOutputShape = outputType.getShape().vec();
      for (TransposeDims dims : transpose_indices) {
        std::swap(newOutputShape[std::get<0>(dims)],
                  newOutputShape[std::get<1>(dims)]);
      }
      auto newOutputType =
          outputType.cloneWith(newOutputShape, outputType.getElementType());
      auto outputTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), newOutputType.getShape(),
          newOutputType.getElementType());

      auto new_pool = rewriter.create<PoolOpType>(
          op.getLoc(), newOutputType, input, outputTensor, kernel_height_attr,
          kernel_width_attr, stride_height_attr, stride_width_attr,
          dilation_height_attr, dilation_width_attr, ceil_mode_attr,
          padding_top_attr, padding_bottom_attr, padding_left_attr,
          padding_right_attr, operand_constraints);

      // Applying the transposes in reverse order to the output will restore the
      // tensor to the original layout
      std::reverse(transpose_indices.begin(), transpose_indices.end());
      Value output =
          generateTransposeSequence(new_pool.getResult(), rewriter,
                                    transpose_indices, operand_constraints);

      // Reverse back so the proper input transposes are generated for the next
      // pool
      std::reverse(transpose_indices.begin(), transpose_indices.end());
      outputs.push_back(output);
    }

    rewriter.replaceOp(op, outputs);
  }

  uint32_t getNumSpatialDims(ttir::PoolingOp op) const {
    uint32_t numSpatialDims = 0;
    for (int64_t dim : op.getWindowDimensions()) {
      if (dim > 1) {
        numSpatialDims++;
      }
    }
    return numSpatialDims;
  }

  LogicalResult
  matchAndRewrite(ttir::PoolingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t numSpatialDims = getNumSpatialDims(op);
    if (numSpatialDims == 2) {
      if (failed(canDecompose2DPoolingOp(op))) {
        return rewriter.notifyMatchFailure(
            op, "2D pooling op with the given attributes is not supported "
                "currently");
      }

      switch (op.getPoolingMethod()) {
      case ttir::PoolingMethod::Max: {
        rewritePool2d<ttir::MaxPool2dOp>(op, adaptor, rewriter);
        return success();
      }
      default: {
        return rewriter.notifyMatchFailure(
            op, "Failed to match pooling method: " +
                    stringifyPoolingMethod(op.getPoolingMethod()));
      }
      }
    }
    return rewriter.notifyMatchFailure(
        op, "No decompositions for a pooling op with " +
                std::to_string(numSpatialDims) + " spatial dimensions");
  }
};

class GetDimensionSizeToConstantConversionPattern
    : public OpConversionPattern<ttir::GetDimensionSizeOp> {
public:
  using OpConversionPattern<ttir::GetDimensionSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GetDimensionSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const RankedTensorType inputTensorType =
        mlir::cast<RankedTensorType>(op.getOperand().getType());

    int64_t dimensionIndex = op.getDimension();

    int32_t dimSize = inputTensorType.getShape()[dimensionIndex];

    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(op.getType());

    mlir::ElementsAttr valueAttr =
        mlir::DenseElementsAttr::get<int>(valueType, dimSize);

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(op, valueType,
                                                            valueAttr);

    return success();
  }
};

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GetDimensionSizeToConstantConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
