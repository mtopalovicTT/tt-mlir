// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SCHEDULER_SCHEDULER_H
#define TTMLIR_SCHEDULER_SCHEDULER_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::scheduler {

// Helper scheduler functions
bool isTTNNOp(mlir::Operation *op);
bool isTTIROp(mlir::Operation *op);
bool isTTShedulableOp(mlir::Operation *op);

class Scheduler {
public:
  // Constructor taking an MLIR Operation (or a module)
  Scheduler(func::FuncOp *root);

  // Copy constructor
  Scheduler(const Scheduler &scheduler);

  virtual ~Scheduler() = default;

  // Method to schedule an operation
  virtual void scheduleOp(mlir::Operation *op) = 0;

  // Method to get the next set of schedulable operations
  virtual llvm::SmallVector<mlir::Operation *> getScheduleableOps() = 0;

  // Method to get the scheduled operations
  virtual llvm::SmallVector<mlir::Operation *> getSchedule() = 0;

  // Method to take a snapshot of the scheduler
  virtual std::unique_ptr<Scheduler> snapshot() = 0;

  // Method to check if there are unscheduled operations
  bool hasUnscheduledOps() const;

  // Method to check if an operation can be scheduled
  bool canSchedule(mlir::Operation *op);

protected:
  // Map of dependencies
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      dependencies;

  // Sets of unscheduled / schedulable / scheduled operations
  llvm::DenseSet<mlir::Operation *> unscheduledOps;
  llvm::DenseSet<mlir::Operation *> schedulableOps;
  llvm::DenseSet<mlir::Operation *> scheduledOps;

  // Operation schedule in order of execution
  llvm::SmallVector<mlir::Operation *> schedule;
};

} // namespace mlir::tt::scheduler

#endif
