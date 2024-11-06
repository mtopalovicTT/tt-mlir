// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SCHEDULER_SCHEDULER_H
#define TTMLIR_SCHEDULER_SCHEDULER_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::scheduler {

class Scheduler {
public:
  // Constructor taking an MLIR Operation (or a module)
  Scheduler(func::FuncOp *root);

  // Copy constructor
  Scheduler(const Scheduler &scheduler);

  // Method to schedule an operation
  void scheduleOp(mlir::Operation *op);

  // Method to check if there are unscheduled operations
  bool hasUnscheduledOps() const;

  // Method to get the next set of schedulable operations
  llvm::SmallVector<mlir::Operation *> getScheduleableOps();

  // Method to get the scheduled operations
  llvm::SmallVector<mlir::Operation *> getSchedule() const;

  // Method to take a snapshot of the scheduler
  std::unique_ptr<Scheduler> snapshot();

  // Method to check if an operation can be scheduled
  bool canSchedule(mlir::Operation *op);

private:
  // Map of dependencies
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      dependencies;

  // Sets of unscheduled / schedulable / scheduled operations
  llvm::DenseSet<mlir::Operation *> unscheduledOps;
  llvm::SmallVector<mlir::Operation *> schedulableOps;
  llvm::DenseSet<mlir::Operation *> scheduledOps;

  // Operation schedule in order of execution
  llvm::SmallVector<mlir::Operation *> schedule;
};

} // namespace mlir::tt::scheduler

#endif
