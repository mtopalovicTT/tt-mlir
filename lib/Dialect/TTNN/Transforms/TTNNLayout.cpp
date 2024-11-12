// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/Utils/OperandConstraints.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNLAYOUT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Default collapse dims for affine map (d0, d1, d2) -> (d0 <> d1, d2)
static const std::array<std::pair<int64_t, int64_t>, 1> g_defaultCollapseDims =
    {{{0, -1}}};

// Default memory space for tensors on host
static const BufferType g_defaultMemorySpaceHost = BufferType::SystemMemory;

// Default memory space for tesnors on device
static const BufferType g_defaultMemorySpaceDevice = BufferType::DRAM;

// Default memory layout for tensors on device
static const TensorMemoryLayout g_defaultMemoryLayout =
    TensorMemoryLayout::Interleaved;

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

inline Location appendInputSuffix(Location loc, int64_t operandIndex) {
  if (isa<NameLoc>(loc)) {
    NameLoc oldLoc = mlir::cast<NameLoc>(loc);
    StringAttr newName = StringAttr::get(
        loc->getContext(), oldLoc.getName().str() + "_in_" +
                               std::to_string(operandIndex) + "_layout");

    return NameLoc::get(newName, oldLoc.getChildLoc());
  }

  return loc;
}

//===----------------------------------------------------------------------===//
// To layout pass
//===----------------------------------------------------------------------===//

class TTNNLayoutTensorTypeConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeConverter(MLIRContext *ctx, GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, deviceGrid](RankedTensorType type) -> Type {
      auto layout = type.getEncoding();
      if (layout) {
        return type;
      }

      std::int64_t deviceGridRank = deviceGrid.getShape().size();

      // Default to single core grid
      auto tensorGrid = GridAttr::get(ctx, deviceGridRank);

      llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
          g_defaultCollapseDims);

      auto newLayout = TensorConfigAttr::get(
          ctx, type.getShape(), type.getElementType(), g_defaultMemorySpaceHost,
          tensorGrid, TensorMemoryLayout::None, collapseDimsRef);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};

class TTNNLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTNNLayoutTensorTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(&converter) {}

  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    if (failed(converter->convertTypes(valueRange.getTypes(), newTypes))) {
      return false;
    }

    bool updated = false;
    for (auto [value, newType] : llvm::zip(valueRange, newTypes)) {
      if (value.getType() != newType) {
        value.setType(newType);
        updated = true;
      }
    }
    return updated;
  }

  bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp) {
      return false;
    }
    SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes(funcOp.getResultTypes());

    for (Type &ty : inputTypes) {
      ty = converter->convertType(ty);
    }
    for (Type &ty : outputTypes) {
      ty = converter->convertType(ty);
    }
    auto newType = rewriter.getType<FunctionType>(inputTypes, outputTypes);
    if (funcOp.getFunctionType() == newType) {
      return false;
    }
    funcOp.setFunctionType(newType);

    Block &entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      entryBlock.getArgument(i).setType(inputTypes[i]);
    }

    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool updated = false;
    SmallVector<Type> operands;
    SmallVector<Type> results;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    updated |= convertFuncType(op, rewriter);
    return updated ? success() : failure();
  }

  const TypeConverter *converter;
};

static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 BufferType desiredBufferType,
                 TensorMemoryLayout desiredMemLayout, bool tiled) {

  // Get type
  auto ty = mlir::cast<RankedTensorType>(input.getType());

  // Get tensor config from the type
  auto tensorConfig = mlir::cast<TensorConfigAttr>(ty.getEncoding());

  // Get buffer type (i.e DRAM/L1 etc)
  auto currBufferType = tensorConfig.getBufferType();

  // Get the current element type (i.e bf16/TileType etc)
  auto currElementType = tensorConfig.getElementType();

  // Get the mem layout attribute (i.e interleaved/sharded or null in case of
  // System)
  auto currMemLayout = tensorConfig.getMemLayout();

  // Get element type that should be used in the new tensor config
  auto desiredElementType =
      tiled ? rewriter.getType<TileType>(ty.getElementType())
            : ty.getElementType();

  if (currBufferType == desiredBufferType &&
      currElementType == desiredElementType &&
      currMemLayout == desiredMemLayout) {
    return std::nullopt;
  }

  auto desiredLayout = rewriter.getAttr<TensorConfigAttr>(
      ty.getShape(), desiredElementType, desiredBufferType,
      tensorConfig.getGrid(), desiredMemLayout, g_defaultCollapseDims);

  tensor::EmptyOp existingEmpty = input.getDefiningOp<tensor::EmptyOp>();
  if (existingEmpty) {
    return rewriter
        .replaceOpWithNewOp<tensor::EmptyOp>(existingEmpty, ty.getShape(),
                                             ty.getElementType(), desiredLayout)
        .getResult();
  }

  ttir::ConstantOp existingConstant = input.getDefiningOp<ttir::ConstantOp>();
  if (existingConstant) {
    return rewriter
        .replaceOpWithNewOp<ttir::ConstantOp>(
            existingConstant,
            mlir::RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                        desiredLayout),
            existingConstant.getValue())
        .getResult();
  }

  tensor::EmptyOp output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  return rewriter
      .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 OperandConstraint operandConstraint) {
  // Find out which memory space we want
  tt::MemorySpace ttDefaultMemSpace =
      utils::toTTMemorySpace(g_defaultMemorySpaceDevice);
  tt::MemorySpace desiredMemorySpace =
      getLegalMemorySpace(operandConstraint, ttDefaultMemSpace);

  // Convert it to TTNN buffer type
  BufferType desiredBufferType = utils::toTTNNBufferType(desiredMemorySpace);

  // Find out which memory layout we want
  tt::TensorMemoryLayout ttMemoryLayout =
      utils::toTTTensorMemoryLayout(g_defaultMemoryLayout);
  tt::TensorMemoryLayout desiredMemoryLayout = getLegalTensorMemoryLayout(
      operandConstraint, desiredMemorySpace, ttMemoryLayout);
  TensorMemoryLayout ttnnMemoryLayout =
      utils::toTTNNTensorMemoryLayout(desiredMemoryLayout);

  // Check if the tensor should be tiled
  bool tiled =
      !bitEnumContainsAny(operandConstraint, OperandConstraint::Scalar);

  return createToLayoutOp(rewriter, loc, input, desiredBufferType,
                          ttnnMemoryLayout, tiled);
}

class TTNNLayoutDPSOperandsRewriter
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
public:
  TTNNLayoutDPSOperandsRewriter(MLIRContext *ctx)
      : OpInterfaceRewritePattern<DestinationStyleOpInterface>(ctx) {}

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<ttir::ToLayoutOp>(op.getOperation())) {
      // Skip the ToLayoutOp itself.
      return failure();
    }

    assert(op->template hasTrait<ttir::TTIROp::Trait>());
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      bool isResult = op.isDpsInit(&operand);

      // TTNN Conv2d moves input, weight, and bias from host to device
      // itself. Inserting the ToLayoutOp on these operands is thus problematic.
      if (mlir::isa<ttir::Conv2dOp>(op.getOperation()) && !isResult) {
        continue;
      }
      auto operandConstraint =
          mlir::cast<OperandConstraintAttr>(
              mlir::cast<ttir::TTIROp>(op.getOperation())
                  .getOperandConstraints()[operand.getOperandNumber()])
              .getValue();
      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      auto desiredLayout =
          createToLayoutOp(rewriter, newLoc, operand.get(), operandConstraint);

      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          if (isResult) {
            // If this is the output operand, update the result type
            op->getResult(0).setType(desiredLayout->getType());
          }
        });
      }
    }

    return modified ? success() : failure();
  }
};

class TTNNLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTNNLayoutFuncReturnRewriter(MLIRContext *ctx)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      // Leave the return values in initMemorySpace, optimizer might decide
      // otherwise

      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      auto layout = createToLayoutOp(
          rewriter, newLoc, operand.get(), BufferType::SystemMemory,
          TensorMemoryLayout::None, false /* tiled */);
      if (layout.has_value()) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operand.getOperandNumber(), *layout); });
        modified = true;
      }
    }
    return modified ? success() : failure();
  }

private:
};

class TTNNLayout : public impl::TTNNLayoutBase<TTNNLayout> {
public:
  using impl::TTNNLayoutBase<TTNNLayout>::TTNNLayoutBase;

  void runOnOperation() final {
    // First add default attribute to all tensors. Example:
    // Given tensor type: tensor<15x10x32xf32>
    // we construct a tensor config attribute with default values:
    // tensor_config<affine_map, grid<1x1>, memref<<15x64>xf32, #system_memory>
    {
      auto device = getCurrentScopeDevice(getOperation());
      assert(device && "Device not found");
      TTNNLayoutTensorTypeConverter typeConverter(&getContext(),
                                                  device.getWorkerGrid());
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNLayoutTensorTypeRewriter>(typeConverter, &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNLayoutDPSOperandsRewriter>(&getContext());
      patterns.add<TTNNLayoutFuncReturnRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.useTopDownTraversal = true;
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttnn
// TODO (milant) We don't need inital memory space and default device memory and
// default device memory layout since its not used anywhere try to move
// caclulateLogicalShardShape to Utils file try to move collapsedLinearAffineMap
// to Utils file remove adding tt.layout
//  then we need to have a pass before ttir to ttnn pass that will add new
//  tensor config attribute only drawback is that we will have to remove check
//  for layout in ttir ops mlirtoflatbuffr.h now has to know about ttnn tensor
//  config attribute
//      we should split all layout specific stuff to separate header