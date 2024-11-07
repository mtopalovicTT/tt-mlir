// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SCHEDULER_PRECEDENCESCHEDULER_H
#define TTMLIR_SCHEDULER_PRECEDENCESCHEDULER_H

#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::scheduler {

class PrecedenceScheduler : public Scheduler {
public:
  // Constructor taking an MLIR Operation (or a module)
  PrecedenceScheduler(func::FuncOp *root);

  // Copy constructor
  PrecedenceScheduler(const PrecedenceScheduler &scheduler)
      : Scheduler(scheduler) {};

  // Method to schedule an operation
  void scheduleOp(mlir::Operation *op) final;

  // Method to get the next set of schedulable operations
  llvm::SmallVector<mlir::Operation *> getScheduleableOps() final;

  // Method to get the scheduled operations
  llvm::SmallVector<mlir::Operation *> getSchedule() final;

  // Method to take a snapshot of the scheduler
  std::unique_ptr<Scheduler> snapshot() final;

private:
  // Map of precedence
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      precedence;
  // Output op of the function
  mlir::Operation *outputOp;

  // DFS schedule construction based on a precedence map
  llvm::DenseSet<mlir::Operation *> visitedOps;
  void constructSchedule(mlir::Operation *op);
};

} // namespace mlir::tt::scheduler

#endif // TTMLIR_SCHEDULER_PRECEDENCESCHEDULER_H
