// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/PrecedenceScheduler.h"

namespace mlir::tt::scheduler {

void PrecedenceScheduler::scheduleOp(mlir::Operation *op) { return; }

llvm::SmallVector<mlir::Operation *> PrecedenceScheduler::getScheduleableOps() {
  return {};
}

llvm::SmallVector<mlir::Operation *> PrecedenceScheduler::getSchedule() {
  return {};
}

std::unique_ptr<Scheduler> PrecedenceScheduler::snapshot() {
  return std::make_unique<PrecedenceScheduler>(*this);
}

} // namespace mlir::tt::scheduler
