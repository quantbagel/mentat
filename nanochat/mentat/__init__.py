"""Mentat: standalone compiled algorithmic kernels inside transformers.

Instead of calling external tools, Mentat compiles algorithms directly into
transformer weights. The computation happens inside the forward pass — no
external round-trip, no tool-use handoff.

Usage:
    from nanochat.mentat import CompiledModule, CompiledBlock
    from nanochat.mentat.adder import BinaryAdder

    # Create a compiled module
    adder = BinaryAdder()

    # Get a frozen transformer block with the algorithm in its weights
    block = CompiledBlock(adder, d_model=768)

    # Feed the block explicit execution-trace embeddings in a standalone demo
"""

from nanochat.mentat.base import CompiledModule, CompiledBlock
from nanochat.mentat.stack_machine import CompiledStackMachine, Instruction, StackState, StepResult
from nanochat.mentat.data import StackTraceExample, generate_trace_example
