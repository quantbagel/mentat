"""Standalone demo: compiled stack machine using the Mentat framework.

Run: python experiments/compiled_stack_machine.py
"""

import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.mentat.base import CompiledBlock
from nanochat.mentat.stack_machine import CompiledStackMachine, Instruction


def expand(vec: torch.Tensor, d_model: int) -> torch.Tensor:
    full = torch.zeros(d_model, dtype=vec.dtype)
    full[: vec.numel()] = vec
    return full


def random_program(machine: CompiledStackMachine, rng: random.Random, max_len: int) -> list[Instruction]:
    length = rng.randint(1, max_len)
    program = []
    for _ in range(length):
        op = rng.choice(machine.op_names())
        arg = rng.randrange(machine.value_domain) if op == machine.OP_PUSH else None
        program.append(Instruction(op, arg))
        if op == machine.OP_HALT:
            break
    if program[-1].op != machine.OP_HALT:
        program.append(Instruction(machine.OP_HALT))
    return program


def execute_compiled(
    block: CompiledBlock,
    machine: CompiledStackMachine,
    program: list[Instruction],
    device: torch.device,
) -> tuple[list[int], bool, int]:
    state = machine.initial_state()
    embeddings = block.compiled_embeddings
    step = machine.step_embedding(embeddings)
    d_model = block.qkv.weight.shape[1]
    outputs = []
    errors = 0
    halted = False

    with torch.no_grad():
        for instruction in program:
            state_vec = expand(machine.encode_state(state), d_model)
            op_vec, arg_vec = machine.encode_instruction(instruction, embeddings)
            x = torch.stack(
                [
                    state_vec,
                    expand(op_vec, d_model),
                    expand(arg_vec, d_model),
                    expand(step, d_model),
                ],
                dim=0,
            ).unsqueeze(0).to(device)
            out = block(x)
            state, result = machine.decode_step(out[0, -1].cpu())
            if result.emit is not None:
                outputs.append(result.emit)
            errors += int(result.error)
            if result.halt:
                halted = True
                break

    return outputs, halted, errors


def execute_reference(machine: CompiledStackMachine, program: list[Instruction]) -> tuple[list[int], bool, int]:
    state = machine.initial_state()
    outputs = []
    errors = 0
    halted = False
    for instruction in program:
        state, result = machine.transition(state, instruction)
        if result.emit is not None:
            outputs.append(result.emit)
        errors += int(result.error)
        if result.halt:
            halted = True
            break
    return outputs, halted, errors


def format_program(program: list[Instruction]) -> str:
    pieces = []
    for instruction in program:
        if instruction.arg is None:
            pieces.append(instruction.op)
        else:
            pieces.append(f"{instruction.op} {instruction.arg}")
    return " | ".join(pieces)


def main():
    rng = random.Random(42)
    device = torch.device("cpu")
    machine = CompiledStackMachine(value_domain=4, max_stack_depth=3)
    block = CompiledBlock(machine, d_model=64).to(device)
    nparams = sum(p.numel() for p in block.parameters())

    print("=" * 60)
    print("  Mentat Compiled Stack Machine Demo")
    print(f"  {nparams:,} frozen parameters, zero training")
    print("  Ops: PUSH, ADD, SUB, MUL, OUTPUT, HALT")
    print("  State: depth<=3, values in Z/4Z")
    print("=" * 60)

    num_programs = 500
    matched = 0
    t0 = time.time()
    sample_program = None
    sample_outputs = None

    for _ in range(num_programs):
        program = random_program(machine, rng, max_len=12)
        compiled = execute_compiled(block, machine, program, device)
        reference = execute_reference(machine, program)
        if compiled == reference:
            matched += 1
            if sample_program is None and compiled[0]:
                sample_program = program
                sample_outputs = compiled

    dt = time.time() - t0
    print(f"  Random programs: {matched}/{num_programs} = {matched/num_programs:.1%}  ({dt:.2f}s)")
    if sample_program is not None and sample_outputs is not None:
        print("  Example:")
        print(f"    {format_program(sample_program)}")
        print(f"    outputs={sample_outputs[0]} halted={sample_outputs[1]} errors={sample_outputs[2]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
