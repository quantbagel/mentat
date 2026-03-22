import random

import torch

from nanochat.mentat.base import CompiledBlock
from nanochat.mentat.stack_machine import CompiledStackMachine, Instruction


def expand(vec, d_model):
    full = torch.zeros(d_model, dtype=vec.dtype)
    full[: vec.numel()] = vec
    return full


def compiled_step(block, machine, state, instruction):
    embeddings = block.compiled_embeddings
    d_model = block.qkv.weight.shape[1]
    x = torch.stack(
        [
            expand(machine.encode_state(state), d_model),
            expand(embeddings[machine.op_token_id(instruction.op)], d_model),
            expand(embeddings[machine.arg_token_id(instruction.arg)], d_model),
            expand(embeddings[machine.step_token_id], d_model),
        ],
        dim=0,
    ).unsqueeze(0)
    with torch.no_grad():
        out = block(x)
    return machine.decode_step(out[0, -1])


def compiled_program(block, machine, program):
    state = machine.initial_state()
    outputs = []
    errors = 0
    halted = False
    for instruction in program:
        state, result = compiled_step(block, machine, state, instruction)
        if result.emit is not None:
            outputs.append(result.emit)
        errors += int(result.error)
        if result.halt:
            halted = True
            break
    return outputs, halted, errors


def reference_program(machine, program):
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


def test_compiled_stack_machine_matches_every_transition():
    machine = CompiledStackMachine(value_domain=4, max_stack_depth=3)
    block = CompiledBlock(machine, d_model=64)

    for state, instruction in machine.transitions():
        expected = machine.transition(state, instruction)
        actual = compiled_step(block, machine, state, instruction)
        assert actual == expected, (state, instruction, actual, expected)


def test_compiled_stack_machine_matches_random_programs():
    machine = CompiledStackMachine(value_domain=4, max_stack_depth=3)
    block = CompiledBlock(machine, d_model=64)
    rng = random.Random(123)

    for _ in range(200):
        program = []
        for _ in range(rng.randint(1, 10)):
            op = rng.choice(machine.op_names())
            arg = rng.randrange(machine.value_domain) if op == machine.OP_PUSH else None
            program.append(Instruction(op, arg))
            if op == machine.OP_HALT:
                break
        if program[-1].op != machine.OP_HALT:
            program.append(Instruction(machine.OP_HALT))
        assert compiled_program(block, machine, program) == reference_program(machine, program)


def test_reference_program_example():
    machine = CompiledStackMachine(value_domain=4, max_stack_depth=3)
    program = [
        Instruction(machine.OP_PUSH, 3),
        Instruction(machine.OP_PUSH, 2),
        Instruction(machine.OP_ADD),
        Instruction(machine.OP_OUTPUT),
        Instruction(machine.OP_PUSH, 3),
        Instruction(machine.OP_MUL),
        Instruction(machine.OP_OUTPUT),
        Instruction(machine.OP_HALT),
    ]

    assert reference_program(machine, program) == ([1, 3], True, 0)
