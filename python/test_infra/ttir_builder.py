# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, func, tensor
import torch

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int]]


@dataclass(frozen=True)
class Golden:
    """
    Dataclass used to store information about golden tensor which will be used
    for comparison with TT device output.

    Each TTIR op should have a matching torch op, and for same inputs, they
    should generate same outputs.
    """

    tensor: torch.Tensor

    # `torch.manual_seed` arg with which tensor was generated. Valid (not None)
    # only for randomly generated tensors, for example args of MLIR function
    # wrapped around user-written op graph. Every other tensor is output of some
    # op from graph.
    seed: int = None

    def __repr__(self) -> str:
        s = f"\nRandom seed: {self.seed}" if self.seed is not None else ""
        s += f"\nGolden tensor:\n{self.tensor}"
        return s


class TTIRBuilder:
    """Builder class providing API for creating TTIR ops."""

    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location

        tt.register_dialect(self._ctx)
        ttir.register_dialect(self._ctx)

        self._seed = 0
        # Dictionary to store Golden for each Operand we encounter in MLIR
        # graph.
        self._goldens: Dict[Operand, Golden] = {}

        # global ID of operations
        self._global_id = -1

        # id to golden map
        self.id_golden_map = {}

    # ----- Public helpers -----

    @property
    def goldens(self) -> Dict:
        return self._goldens

    def get_next_global_id(self) -> int:
        self._global_id += 1
        return self._global_id

    def print_goldens(self) -> None:
        """
        Prints saved operands and their respective goldens in descriptive form
        which follows SSA ordering from MLIR graph.
        """
        i = 0
        for operand, golden in self._goldens.items():
            operand_name = self._get_name(operand)

            if self._operand_is_mlir_func_arg(operand):
                print(f"Func arg: {operand_name}", golden, "\n")
            else:
                print(f"%{i}: {operand_name}", golden, "\n")
                i += 1

    def get_shape(self, input: Operand) -> Shape:
        """Retrieves shape of operand which is expected to be a shaped type."""
        return self._get_type(input).shape

    def generate_and_store_random_golden(self, operand: Operand) -> None:
        """
        Generates random tensor of `operand`s shape, assigns it to a golden,
        and maps `operand` to that golden.
        """
        seed = self._get_seed()
        random_tensor = self._generate_random_tensor(self.get_shape(operand), seed)
        golden = Golden(random_tensor, seed)
        self._store_golden(operand, golden)

    def generate_input_golden(self, operand: Operand, index: int) -> None:
        """
        Generates random tensor of `input`s shape, assigns it to a golden,
        and maps `input` to that golden.
        """
        seed = self._get_seed()
        random_tensor = self._generate_random_tensor(self.get_shape(operand), seed)
        golden = Golden(random_tensor, seed)
        self.id_golden_map[f"input_{str(index)}"] = golden
        self._store_golden(operand, golden)

    # ----- Private helpers -----

    @staticmethod
    def _get_name(operand: Operand) -> str:
        """Retrieves descriptive operand name."""
        # Try to call get_name() if it exists, otherwise return operand.name.
        name = getattr(operand, "get_name", lambda: None)() or getattr(
            operand, "name", None
        )
        assert name is not None, (
            f"Couldn't retrieve name for operand {operand}. Check if this "
            f"operand type is properly supported."
        )
        return name

    @staticmethod
    def _operand_is_mlir_func_arg(operand: Operand) -> bool:
        """Checks if operand is an argument of surrounding MLIR function."""
        return isinstance(operand, BlockArgument) and "arg" in TTIRBuilder._get_name(
            operand
        )

    def _get_seed(self) -> int:
        """Monotonically increasing seed for reproducibility."""
        seed = self._seed
        self._seed += 1
        return seed

    @staticmethod
    def _generate_random_tensor(shape: Shape, seed: int) -> torch.Tensor:
        """
        Generates random tensor of shape `shape`, using `seed` to seed torch
        random generator.
        """
        return torch.randn(shape, generator=torch.manual_seed(seed))

    def _get_golden(self, operand: Operand) -> Golden:
        """Retrieves stored golden for `operand`."""
        golden = self._goldens.get(operand)
        assert golden is not None, f"Expected to have a golden stored for {operand}"
        return golden

    def _store_golden(self, operand: Operand, golden: Golden) -> None:
        """Maps `operand` to `golden`."""
        assert (
            self._goldens.get(operand) == None
        ), f"Golden for {operand} already exists."
        self._goldens[operand] = golden

    def _override_golden(self, operand: Operand, golden: Golden) -> None:
        """
        Overrides existing golden for `operand`.

        Used to override randomly generated goldens for empty tensors which are
        used as outputs of TTIR ops with golden for that TIIR op.
        """
        assert (
            self._goldens.get(operand) is not None
        ), f"Expected golden for {operand} to already exist before overriding it."
        self._goldens[operand] = golden

    def _get_golden_tensor(self, operand: Operand) -> torch.Tensor:
        return self._get_golden(operand).tensor

    def _get_operand_constraint_attr(
        self,
        num_operands: int,
        operand_constraints: Optional[List[tt.OperandConstraint]] = None,
    ) -> tt.ir.OperandConstraintAttr:
        """
        Helper method to prepack operand constraints given as a list of enums
        to a list of tt.ir.OperandConstraintAttr and wrap that list in an
        tt.ir.OperandConstraintAttr.

        If no `operand_constraints` are passed, `tt.OperandConstraint.Any` will
        be used for each operand.
        """
        operand_constraints = (
            operand_constraints
            if operand_constraints is not None
            else [tt.OperandConstraint.Any for _ in range(num_operands)]
        )

        return tt.ir.OperandConstraintAttr.get(
            self._ctx,
            [
                tt.ir.OperandConstraintAttr.get(self._ctx, operand_constraint)
                for operand_constraint in operand_constraints
            ],
        )

    @property
    def _default_dtype(self) -> Type:
        return F32Type.get(self._ctx)

    def _get_type(self, input: Operand):
        """
        Helper method which retrieves underlying mlir Type of Operand based on
        which concrete type it is.

        We always expect it to be a RankedTensorType.
        """
        if isinstance(input, Value):
            typ = input.type
        elif isinstance(input, OpView):
            typ = input.operation.result.type
        elif isinstance(input, Operation):
            typ = input.result.type
        else:
            raise TypeError(f"Invalid input {type(input)}")

        assert isinstance(typ, RankedTensorType), "Only ranked tensors are supported"

        return typ

    # ----- Utility factories -----

    def ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        """Convenience wrapper constructing `RankedTensorType`."""
        dtype = data_type if data_type is not None else self._default_dtype
        with self._ctx, self._loc:
            return RankedTensorType.get(shape, dtype, encoding)

    def empty(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
    ) -> OpView:
        """Convenience wrapper constructing `tensor.EmptyOp`."""
        dtype = data_type if data_type is not None else self._default_dtype
        with self._ctx, self._loc:
            op = tensor.EmptyOp(shape, dtype)

            self.generate_and_store_random_golden(op)

            return op

    # ----- TTIR op factories -----
    def eltwise_proxy(
        self, op_golden_function, op_ttir_function, inputs: List[Operand]
    ) -> OpView:
        with self._ctx, self._loc:
            output = self.empty(self.get_shape(inputs[0]))

            id = self.get_next_global_id()

            op = op_ttir_function(
                [self._get_type(output)],
                inputs,
                [output],
                self._get_operand_constraint_attr(3),
                loc=Location.name(str(id)),
            )

            goldens = []
            for input in inputs:
                goldens.append(self._get_golden_tensor(input))

            golden = Golden(op_golden_function(*goldens))
            self.id_golden_map[str(id)] = golden
            self._store_golden(op, golden)
            self._override_golden(output, golden)

            return op

    def exp(self, in0: Operand, golden_func: Callable = torch.exp) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.ExpOp, [in0])

    def abs(self, in0: Operand, golden_func: Callable = torch.abs) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.AbsOp, [in0])

    def logical_not(
        self, in0: Operand, golden_func: Callable = torch.logical_not
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.LogicalNotOp, [in0])

    def neg(self, in0: Operand, golden_func: Callable = torch.neg) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.NegOp, [in0])

    def relu(self, in0: Operand, golden_func: Callable = torch.relu) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.ReluOp, [in0])

    def sqrt(self, in0: Operand, golden_func: Callable = torch.sqrt) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.SqrtOp, [in0])

    def rsqrt(self, in0: Operand, golden_func: Callable = torch.rsqrt) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.RsqrtOp, [in0])

    def sigmoid(self, in0: Operand, golden_func: Callable = torch.sigmoid) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.SigmoidOp, [in0])

    def reciprocal(
        self, in0: Operand, golden_func: Callable = torch.reciprocal
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.ReciprocalOp, [in0])

    def add(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.add
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.AddOp, [in0, in1])

    def multiply(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.multiply
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.MultiplyOp, [in0, in1])

    def logical_and(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.logical_and
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.LogicalAndOp, [in0, in1])

    def logical_or(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.logical_or
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.LogicalOrOp, [in0, in1])

    def subtract(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.subtract
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.SubtractOp, [in0, in1])

    def eq(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.eq
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.EqualOp, [in0, in1])

    def ne(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.ne
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.NotEqualOp, [in0, in1])

    def ge(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.ge
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.GreaterEqualOp, [in0, in1])

    def gt(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.gt
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.GreaterThanOp, [in0, in1])

    def le(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.le
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.LessEqualOp, [in0, in1])

    def lt(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.lt
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.LessThanOp, [in0, in1])

    def div(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.div
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.DivOp, [in0, in1])

    def maximum(
        self, in0: Operand, in1: Operand, golden_func: Callable = torch.maximum
    ) -> OpView:
        return self.eltwise_proxy(golden_func, ttir.MaximumOp, [in0, in1])
