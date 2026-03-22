"""Synthetic corpus generation for Mentat public reimplementation work."""

from __future__ import annotations

import random
from dataclasses import dataclass

from nanochat.mentat.stack_machine import CompiledStackMachine, Instruction, StackState


TEST_RANDOM_SEED_OFFSET = 10_000_000


@dataclass(frozen=True)
class StackTraceExample:
    """A single program plus its exact execution trace."""

    program: list[Instruction]
    text: str
    outputs: list[int]
    halted: bool
    errors: int


def stack_contents(state: StackState) -> list[int]:
    """Return the live stack, top first."""
    return list(state.slots[: state.depth])


def render_instruction(instruction: Instruction) -> str:
    return instruction.op if instruction.arg is None else f"{instruction.op} {instruction.arg}"


def generate_program(
    machine: CompiledStackMachine,
    rng: random.Random,
    min_len: int = 4,
    max_len: int = 18,
    invalid_rate: float = 0.12,
    halt_rate: float = 0.08,
) -> list[Instruction]:
    """Generate a deterministic but varied stack-machine program."""
    target_len = rng.randint(min_len, max_len)
    program: list[Instruction] = []
    state = machine.initial_state()
    outputs_seen = 0

    for step_idx in range(target_len):
        if step_idx >= min_len and (rng.random() < halt_rate or outputs_seen >= 2):
            program.append(Instruction(machine.OP_HALT))
            break

        invalid = rng.random() < invalid_rate
        candidates: list[str] = []
        if state.depth == 0:
            candidates.extend([machine.OP_PUSH, machine.OP_PUSH, machine.OP_PUSH, machine.OP_OUTPUT])
            if invalid:
                candidates.extend([machine.OP_ADD, machine.OP_SUB, machine.OP_MUL])
        elif state.depth == 1:
            candidates.extend([machine.OP_PUSH, machine.OP_PUSH, machine.OP_OUTPUT, machine.OP_OUTPUT])
            candidates.extend([machine.OP_ADD, machine.OP_SUB, machine.OP_MUL] if invalid else [machine.OP_PUSH])
        else:
            candidates.extend([machine.OP_ADD, machine.OP_SUB, machine.OP_MUL, machine.OP_OUTPUT, machine.OP_PUSH])

        if state.depth >= machine.max_stack_depth:
            candidates = [op for op in candidates if op != machine.OP_PUSH] or [machine.OP_OUTPUT]

        op = rng.choice(candidates)
        arg = rng.randrange(machine.value_domain) if op == machine.OP_PUSH else None
        instruction = Instruction(op, arg)
        program.append(instruction)
        state, result = machine.transition(state, instruction)
        outputs_seen += int(result.emit is not None)

    if not program or program[-1].op != machine.OP_HALT:
        program.append(Instruction(machine.OP_HALT))
    return program


def render_trace_document(
    machine: CompiledStackMachine,
    program: list[Instruction],
) -> StackTraceExample:
    """Render one exact execution-trace document."""
    state = machine.initial_state()
    outputs: list[int] = []
    halted = False
    errors = 0
    lines = [
        "Mentat stack machine execution example.",
        f"machine: modulo={machine.value_domain} max_stack_depth={machine.max_stack_depth}",
        "",
        "program:",
    ]

    for pc, instruction in enumerate(program):
        lines.append(f"{pc}: {render_instruction(instruction)}")

    lines.extend(["", "trace:"])
    for pc, instruction in enumerate(program):
        before = stack_contents(state)
        state, result = machine.transition(state, instruction)
        after = stack_contents(state)
        emit = "none" if result.emit is None else str(result.emit)
        lines.append(
            " | ".join(
                [
                    f"step={pc}",
                    f"pc={pc}",
                    f"op={instruction.op}",
                    f"arg={'none' if instruction.arg is None else instruction.arg}",
                    f"before={before}",
                    f"after={after}",
                    f"emit={emit}",
                    f"error={int(result.error)}",
                    f"halt={int(result.halt)}",
                ]
            )
        )
        if result.emit is not None:
            outputs.append(result.emit)
        errors += int(result.error)
        if result.halt:
            halted = True
            break

    lines.extend(
        [
            "",
            "final:",
            f"outputs={outputs}",
            f"halted={int(halted)}",
            f"errors={errors}",
        ]
    )
    return StackTraceExample(program=program, text="\n".join(lines), outputs=outputs, halted=halted, errors=errors)


def generate_trace_example(
    index: int,
    split: str,
    value_domain: int = 4,
    max_stack_depth: int = 3,
    min_program_len: int = 4,
    max_program_len: int = 18,
) -> StackTraceExample:
    """Deterministically generate one train/val example."""
    assert split in {"train", "val"}, f"invalid split: {split}"
    seed = index if split == "train" else TEST_RANDOM_SEED_OFFSET + index
    rng = random.Random(seed)
    machine = CompiledStackMachine(value_domain=value_domain, max_stack_depth=max_stack_depth)
    program = generate_program(
        machine,
        rng,
        min_len=min_program_len,
        max_len=max_program_len,
    )
    return render_trace_document(machine, program)
