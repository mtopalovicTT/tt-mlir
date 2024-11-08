// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"

namespace mlir::tt::ttnn {

struct OpNode {
  // Operation that is wrapped within OpNode struct.
  //
  Operation *op;

  // Layout of the output tensor of the op.
  //
  tt::LayoutAttr layout;

  // Minimum L1 memory usage required for scheduling the op
  // given the layouts of all the ops that are already scheduled.
  //
  uint64_t maxL1Usage;
};

class L1InterleavedPolicy : public MemoryLayoutAnalysisPolicy {
public:
  L1InterleavedPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalLayouts,
                                   schedule, usableL1CacheSize) {}

  void run() final;

private:
  // Fetch op's DRAM layout from legalLayouts
  bool hasDRAMLayout(Operation *op);
  tt::LayoutAttr getDRAMLayout(Operation *op);

  // Fetch op's L1 Interleaved layout from legalLayouts
  bool hasL1InterleavedLayout(Operation *op);
  tt::LayoutAttr getL1InterleavedLayout(Operation *op);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H
