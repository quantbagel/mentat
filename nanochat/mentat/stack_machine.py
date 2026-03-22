"""Compiled stack-machine transition kernel for Mentat.

Phase 1 is intentionally modest: the transformer block executes the exact
transition function of a small stack machine with multiple instruction types.
The program counter remains external to the block, while the stack state update
itself happens inside frozen transformer weights.

The machine semantics are:
  - Fixed maximum stack depth
  - Small integer domain with arithmetic modulo `value_domain`
  - Instructions: PUSH, ADD, SUB, MUL, OUTPUT, HALT
  - Invalid operations (stack underflow / overflow) become no-ops and raise an
    error flag in the step output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from nanochat.mentat.base import CompiledModule, CompiledWeights


@dataclass(frozen=True)
class StackState:
    """Canonical stack state used by the reference executor and encoder."""

    depth: int
    slots: tuple[int, ...]

    def __post_init__(self):
        if self.depth < 0 or self.depth > len(self.slots):
            raise ValueError(f"invalid depth {self.depth} for {len(self.slots)} slots")


@dataclass(frozen=True)
class StepResult:
    """Auxiliary outputs produced by a single machine step."""

    emit: int | None = None
    halt: bool = False
    error: bool = False


@dataclass(frozen=True)
class Instruction:
    """Single stack-machine instruction."""

    op: str
    arg: int | None = None


class CompiledStackMachine(CompiledModule):
    """Frozen transformer block that executes a small stack-machine step."""

    OP_PUSH = "PUSH"
    OP_ADD = "ADD"
    OP_SUB = "SUB"
    OP_MUL = "MUL"
    OP_OUTPUT = "OUTPUT"
    OP_HALT = "HALT"
    OPS = (OP_PUSH, OP_ADD, OP_SUB, OP_MUL, OP_OUTPUT, OP_HALT)

    TYPE_STATE = 0
    TYPE_OP = 1
    TYPE_ARG = 2
    TYPE_STEP = 3

    def __init__(self, value_domain: int = 4, max_stack_depth: int = 3):
        if value_domain < 2:
            raise ValueError("value_domain must be at least 2")
        if max_stack_depth < 2:
            raise ValueError("max_stack_depth must be at least 2")
        self.value_domain = value_domain
        self.max_stack_depth = max_stack_depth

        self.state_dims = (max_stack_depth + 1) + max_stack_depth * value_domain
        self.op_offset = self.state_dims
        self.arg_offset = self.op_offset + len(self.OPS)
        self.type_offset = self.arg_offset + (value_domain + 1)  # includes ARG_NONE
        self.output_state_offset = self.type_offset + 4
        self.emit_flag_dim = self.output_state_offset + self.state_dims
        self.halt_flag_dim = self.emit_flag_dim + 1
        self.error_flag_dim = self.halt_flag_dim + 1
        self.emit_value_offset = self.error_flag_dim + 1
        self._compiled_dims = self.emit_value_offset + value_domain
        self._step_token_id = 1024
        self._op_token_ids = {op: 1100 + i for i, op in enumerate(self.OPS)}
        self._arg_none_token_id = 1200
        self._arg_token_ids = {value: 1201 + value for value in range(value_domain)}

    @property
    def compiled_dims(self) -> int:
        return self._compiled_dims

    @property
    def window_size(self) -> int:
        # Each transition reads [state, op, arg, step].
        return 4

    @property
    def n_active_heads(self) -> int:
        return self.state_dims + len(self.OPS) + (self.value_domain + 1)

    @property
    def ffn_hidden(self) -> int:
        return len(self.transitions())

    @property
    def step_token_id(self) -> int:
        return self._step_token_id

    def initial_state(self) -> StackState:
        return StackState(depth=0, slots=(0,) * self.max_stack_depth)

    def op_names(self) -> tuple[str, ...]:
        return self.OPS

    def transitions(self) -> list[tuple[StackState, Instruction]]:
        """Enumerate every valid input combination for the transition table."""
        transitions = []
        for state in self.iter_states():
            for op in self.OPS:
                if op == self.OP_PUSH:
                    for value in range(self.value_domain):
                        transitions.append((state, Instruction(op, value)))
                else:
                    transitions.append((state, Instruction(op)))
        return transitions

    def iter_states(self) -> Iterable[StackState]:
        """Enumerate canonical states with zero-filled unused slots."""
        slots = [0] * self.max_stack_depth

        def rec(depth: int, pos: int):
            if pos == depth:
                yield StackState(depth=depth, slots=tuple(slots))
                return
            for value in range(self.value_domain):
                slots[pos] = value
                yield from rec(depth, pos + 1)
            slots[pos] = 0

        yield self.initial_state()
        for depth in range(1, self.max_stack_depth + 1):
            yield from rec(depth, 0)

    def compile(self, d_model: int, n_heads: int) -> CompiledWeights:
        if d_model < self.compiled_dims:
            raise ValueError(f"d_model={d_model} must be >= compiled_dims={self.compiled_dims}")
        if n_heads < self.n_active_heads:
            raise ValueError(f"need at least {self.n_active_heads} heads, got {n_heads}")

        score_scale = 12.0
        match_scale = 8.0

        embeddings = self._compile_embeddings()
        qkv = torch.zeros(3 * d_model, d_model)
        out_proj = torch.zeros(d_model, d_model)

        # Shared type-selective key layout:
        #   state -> [ +S,  0]
        #   op    -> [  0, +S]
        #   arg   -> [ -S, -S]
        for head in range(self.n_active_heads):
            k_row = d_model + 2 * head
            q_row = 2 * head
            self._configure_head_key(qkv, k_row, score_scale)

            if head < self.state_dims:
                self._configure_state_head(qkv, out_proj, d_model, head, head, score_scale)
            elif head < self.state_dims + len(self.OPS):
                src = self.op_offset + (head - self.state_dims)
                self._configure_op_head(qkv, out_proj, d_model, head, src, score_scale)
            else:
                src = self.arg_offset + (head - self.state_dims - len(self.OPS))
                self._configure_arg_head(qkv, out_proj, d_model, head, src, score_scale)

        transitions = self.transitions()
        hidden = len(transitions)
        w1 = torch.zeros(hidden, d_model)
        b1 = torch.zeros(hidden)
        w2 = torch.zeros(d_model, hidden)
        active_value = 0.5 * match_scale
        down_scale = 1.0 / active_value

        for i, (state, instruction) in enumerate(transitions):
            required_dims = self._required_dims(state, instruction)
            for dim in required_dims:
                w1[i, dim] = match_scale
            b1[i] = -(len(required_dims) - 0.5) * match_scale

            next_state, result = self.transition(state, instruction)
            for dim in self._output_state_dims(next_state):
                w2[dim, i] = down_scale
            if result.emit is not None:
                w2[self.emit_flag_dim, i] = down_scale
                w2[self.emit_value_offset + result.emit, i] = down_scale
            if result.halt:
                w2[self.halt_flag_dim, i] = down_scale
            if result.error:
                w2[self.error_flag_dim, i] = down_scale

        return CompiledWeights(
            qkv=qkv,
            out_proj=out_proj,
            ffn_up_w=w1,
            ffn_up_b=b1,
            ffn_down_w=w2,
            ffn_down_b=torch.zeros(d_model),
            embeddings=embeddings,
            head_weights={},
        )

    def format_trace(self, program: Sequence[Instruction]) -> list[int]:
        """Return the discrete instruction/step token skeleton for a program."""
        tokens = []
        for instruction in program:
            tokens.extend([
                self.op_token_id(instruction.op),
                self.arg_token_id(instruction.arg),
                self.step_token_id,
            ])
        return tokens

    def encode_state(self, state: StackState) -> torch.Tensor:
        """Encode a canonical stack state into the compiled subspace."""
        vec = torch.zeros(self.compiled_dims)
        for dim in self._state_feature_dims(state):
            vec[dim] = 1.0
        vec[self.type_offset + self.TYPE_STATE] = 1.0
        return vec

    def op_token_id(self, op: str) -> int:
        return self._op_token_ids[op]

    def arg_token_id(self, arg: int | None) -> int:
        return self._arg_none_token_id if arg is None else self._arg_token_ids[arg]

    def encode_instruction(self, instruction: Instruction, embeddings: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            embeddings[self.op_token_id(instruction.op)].clone(),
            embeddings[self.arg_token_id(instruction.arg)].clone(),
        )

    def step_embedding(self, embeddings: dict[int, torch.Tensor]) -> torch.Tensor:
        return embeddings[self.step_token_id].clone()

    def decode_step(self, vec: torch.Tensor) -> tuple[StackState, StepResult]:
        """Decode the block output into the next canonical state and side effects."""
        state_vec = vec[self.output_state_offset : self.output_state_offset + self.state_dims]
        depth = int(torch.argmax(state_vec[: self.max_stack_depth + 1]).item())
        slots = []
        for slot_idx in range(self.max_stack_depth):
            start = self._output_slot_offset(slot_idx)
            end = start + self.value_domain
            slots.append(int(torch.argmax(vec[start:end]).item()))
        state = StackState(depth=depth, slots=tuple(slots))
        emit = None
        if vec[self.emit_flag_dim].item() > 0.5:
            emit = int(torch.argmax(vec[self.emit_value_offset : self.emit_value_offset + self.value_domain]).item())
        result = StepResult(
            emit=emit,
            halt=vec[self.halt_flag_dim].item() > 0.5,
            error=vec[self.error_flag_dim].item() > 0.5,
        )
        return state, result

    def execute_reference(self, program: Sequence[Instruction]) -> tuple[list[int], bool]:
        """Run the pure-Python stack machine for comparison/testing."""
        state = self.initial_state()
        outputs = []
        halted = False
        for instruction in program:
            state, result = self.transition(state, instruction)
            if result.emit is not None:
                outputs.append(result.emit)
            if result.halt:
                halted = True
                break
        return outputs, halted

    def transition(self, state: StackState, instruction: Instruction) -> tuple[StackState, StepResult]:
        """Pure-Python semantics mirrored exactly by the compiled weights."""
        op = instruction.op
        slots = list(state.slots)
        depth = state.depth
        error = False
        emit = None
        halt = False

        if op == self.OP_PUSH:
            if instruction.arg is None or not (0 <= instruction.arg < self.value_domain):
                raise ValueError(f"invalid PUSH arg: {instruction.arg}")
            if depth >= self.max_stack_depth:
                error = True
            else:
                slots = [instruction.arg, *slots[:-1]]
                depth += 1
        elif op in (self.OP_ADD, self.OP_SUB, self.OP_MUL):
            if depth < 2:
                error = True
            else:
                lhs, rhs = slots[1], slots[0]
                if op == self.OP_ADD:
                    value = (lhs + rhs) % self.value_domain
                elif op == self.OP_SUB:
                    value = (lhs - rhs) % self.value_domain
                else:
                    value = (lhs * rhs) % self.value_domain
                slots = [value, *slots[2:], 0]
                depth -= 1
        elif op == self.OP_OUTPUT:
            if depth < 1:
                error = True
            else:
                emit = slots[0]
        elif op == self.OP_HALT:
            halt = True
        else:
            raise ValueError(f"unknown op: {op}")

        for i in range(depth, self.max_stack_depth):
            slots[i] = 0
        return StackState(depth=depth, slots=tuple(slots)), StepResult(emit=emit, halt=halt, error=error)

    def _compile_embeddings(self) -> dict[int, torch.Tensor]:
        embeddings: dict[int, torch.Tensor] = {}

        step = torch.zeros(self.compiled_dims)
        step[self.type_offset + self.TYPE_STEP] = 1.0
        embeddings[self.step_token_id] = step

        for op_idx, op in enumerate(self.OPS):
            vec = torch.zeros(self.compiled_dims)
            vec[self.op_offset + op_idx] = 1.0
            vec[self.type_offset + self.TYPE_OP] = 1.0
            embeddings[self.op_token_id(op)] = vec

        none_arg = torch.zeros(self.compiled_dims)
        none_arg[self.arg_offset] = 1.0
        none_arg[self.type_offset + self.TYPE_ARG] = 1.0
        embeddings[self._arg_none_token_id] = none_arg
        for value in range(self.value_domain):
            vec = torch.zeros(self.compiled_dims)
            vec[self.arg_offset + 1 + value] = 1.0
            vec[self.type_offset + self.TYPE_ARG] = 1.0
            embeddings[self._arg_token_ids[value]] = vec

        return embeddings

    def _configure_head_key(self, qkv: torch.Tensor, row: int, scale: float) -> None:
        qkv[row, self.type_offset + self.TYPE_STATE] = scale
        qkv[row, self.type_offset + self.TYPE_ARG] = -scale
        qkv[row + 1, self.type_offset + self.TYPE_OP] = scale
        qkv[row + 1, self.type_offset + self.TYPE_ARG] = -scale

    def _configure_state_head(
        self, qkv: torch.Tensor, out_proj: torch.Tensor, d_model: int, head: int, src_dim: int, scale: float
    ) -> None:
        q_row = 2 * head
        v_row = 2 * d_model + 2 * head
        qkv[q_row, self.type_offset + self.TYPE_STEP] = scale
        qkv[v_row, src_dim] = 1.0
        out_proj[src_dim, 2 * head] = 1.0

    def _configure_op_head(
        self, qkv: torch.Tensor, out_proj: torch.Tensor, d_model: int, head: int, src_dim: int, scale: float
    ) -> None:
        q_row = 2 * head
        v_row = 2 * d_model + 2 * head
        qkv[q_row + 1, self.type_offset + self.TYPE_STEP] = scale
        qkv[v_row, src_dim] = 1.0
        out_proj[src_dim, 2 * head] = 1.0

    def _configure_arg_head(
        self, qkv: torch.Tensor, out_proj: torch.Tensor, d_model: int, head: int, src_dim: int, scale: float
    ) -> None:
        q_row = 2 * head
        v_row = 2 * d_model + 2 * head
        qkv[q_row, self.type_offset + self.TYPE_STEP] = -scale
        qkv[q_row + 1, self.type_offset + self.TYPE_STEP] = -scale
        qkv[v_row, src_dim] = 1.0
        out_proj[src_dim, 2 * head] = 1.0

    def _required_dims(self, state: StackState, instruction: Instruction) -> list[int]:
        required = self._state_feature_dims(state)
        required.append(self.op_offset + self.OPS.index(instruction.op))
        required.append(self.arg_dim(instruction.arg))
        return required

    def _state_feature_dims(self, state: StackState) -> list[int]:
        dims = [state.depth]
        for slot_idx, value in enumerate(state.slots):
            dims.append(self._slot_offset(slot_idx) + value)
        return dims

    def _slot_offset(self, slot_idx: int) -> int:
        return (self.max_stack_depth + 1) + slot_idx * self.value_domain

    def _output_state_dims(self, state: StackState) -> list[int]:
        dims = [self.output_state_offset + state.depth]
        for slot_idx, value in enumerate(state.slots):
            dims.append(self._output_slot_offset(slot_idx) + value)
        return dims

    def _output_slot_offset(self, slot_idx: int) -> int:
        return self.output_state_offset + (self.max_stack_depth + 1) + slot_idx * self.value_domain

    def arg_dim(self, arg: int | None) -> int:
        if arg is None:
            return self.arg_offset
        if not (0 <= arg < self.value_domain):
            raise ValueError(f"arg {arg} out of range for value_domain={self.value_domain}")
        return self.arg_offset + 1 + arg
