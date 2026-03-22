# Mentat: Compiled Computation Inside Transformers

> *"The real question is not whether a model can talk about computation, or even
> access it through tools. The real question is whether it can execute computation
> internally."* — [Percepta, "Can LLMs Be Computers?"](https://www.percepta.ai/blog/can-llms-be-computers)

**Mentat** is an open-source public reimplementation effort for compiling
algorithms directly into transformer weights. Instead of calling external tools,
the model *is* the tool: computation happens inside the forward pass, as tokens,
with no external round-trip.

Current scope:
- Standalone compiled kernels and experiment harnesses
- Public, inferential reimplementation only
- No exact claim to Percepta internals
- No NanoChat residual-stream integration for now

## The Problem

LLMs can reason about algorithms but can't reliably execute them. The standard
fix — tool use — means computation happens *outside* the model. Mentat takes
the approach from [Percepta's blog](https://www.percepta.ai/blog/can-llms-be-computers)
and makes it modular and open:

| Approach | Where computation happens | Generalizes? |
|----------|--------------------------|--------------|
| Training | Learned weights (fragile) | Only within training distribution |
| Tool use | External interpreter | Yes, but opaque to the model |
| **Mentat** | **Compiled weights (frozen)** | **Yes, by construction** |

## Current Scope

Mentat currently targets the smallest public unit we can validate directly:
standalone compiled transition kernels. The model block is frozen, the token
schema is explicit, and correctness is checked against a reference interpreter.

Deferred work:
- NanoChat trunk integration / residual-stream plumbing
- In-place weight injection into the main GPT
- HullKVCache
- WASM frontend and full interpreter control flow

## How It Works

### 1. Compiled Modules

An algorithm is expressed as a **step function** with constant lookback distances:

```python
from nanochat.mentat import CompiledModule

class BinaryAdder(CompiledModule):
    compiled_dims = 10   # dims used by the compiled subspace
    window_size = 3      # attention lookback window
    n_active_heads = 3   # type-selective 2D heads
    ffn_hidden = 16      # truth-table indicators

    def compile(self, d_model, n_heads):
        # Hand-craft Q/K/V weights for type-selective attention
        # Hand-craft FFN weights for the truth table
        return CompiledWeights(...)
```

### 2. The Execution Trace

For binary addition of `1011 + 0110`:
```
Input:  BOS 1 0 1 1 + 0 1 1 0 =
Trace:  R00 A1 BA0 [R10] A0 BA1 [R10] A1 BA1 [R01] A1 BA0 [R00] A0 BA0 [R01] EOS
        ↑carry=0  ↑compiled    ↑compiled     ↑compiled     ↑compiled    ↑compiled
```

Every `[R__]` token is computed by the frozen compiled layer reading the 3 preceding
tokens.

## Results

- `compiled_adder.py` verifies exact binary addition at 8/16/32/64 bits with a frozen compiled block.
- `compiled_stack_machine.py` verifies arbitrary random programs over `PUSH`, `ADD`, `SUB`, `MUL`, `OUTPUT`, `HALT` against a reference interpreter.

## Implementing Your Own Module

Any algorithm that can be expressed as a **step function with constant lookbacks**
can be compiled:

1. **Define typed tokens** for your execution trace
2. **Set Q/K weights** so attention heads select tokens by type
3. **Set V weights** to extract the relevant payload (bit values, state, etc.)
4. **Set FFN weights** to implement the truth table / lookup table
5. **Set output head rows** to map results to the correct output token

See `nanochat/mentat/adder.py` for the complete reference implementation.

### Candidates for Compilation

| Algorithm | Step function | Lookback | Complexity |
|-----------|--------------|----------|------------|
| Binary addition | full adder (XOR + MAJ) | 3 tokens | ✅ Done |
| Stack machine core | stack transition + ALU op | 4 tokens | ✅ Phase 1 |
| Multiplication | multiply-accumulate | 3-5 tokens | Medium |
| Constraint propagation | check row/col/box | variable | Hard |
| Finite state machines | state transition | 2 tokens | Easy |
| Sorting networks | compare-and-swap | 2 tokens | Easy |

## Quick Start

```bash
# Run the compiled adder demo
python experiments/compiled_adder.py

# Run the compiled stack-machine demo
python experiments/compiled_stack_machine.py

# Build a Mentat synthetic corpus + tokenizer + first training run
bash runs/mentat_stack.sh

# Output:
#   8-bit: 500/500 = 100.0%
#  16-bit: 500/500 = 100.0%
#  32-bit: 500/500 = 100.0%
#  64-bit: 500/500 = 100.0%
#   stack machine: 500/500 = 100.0%
```

## File Structure

```
nanochat/mentat/
├── __init__.py       # Public API
├── base.py           # CompiledModule and CompiledBlock
├── adder.py          # Binary full-adder (reference implementation)
├── inject.py         # Experimental future integration path
└── stack_machine.py  # Fixed-depth stack-machine transition kernel

experiments/
├── compiled_adder.py          # Standalone demo
└── compiled_stack_machine.py  # Standalone stack-machine demo

scripts/
└── mentat_data.py             # Mentat synthetic parquet corpus generator

runs/
└── mentat_stack.sh            # H100 bootstrap: data -> tokenizer -> training
```

## Roadmap

1. Internal program counter and fetch
2. Branch/control-flow instructions
3. Locals and linear memory
4. Small WASM subset compiler
5. Cache acceleration work

## Training Bootstrap

The current training path does not touch nanochat's main residual stream. It:

1. Generates a synthetic text corpus of exact stack-machine traces
2. Writes that corpus as parquet shards compatible with nanochat's normal tooling
3. Trains a tokenizer on that corpus
4. Runs `scripts.base_train` against the Mentat parquet dataset

The active dataset path can be overridden with `NANOCHAT_DATA_DIR`, which lets
Mentat reuse `tok_train.py` and `base_train.py` unchanged.

## Credits

- Architecture inspired by [Percepta's "Can LLMs Be Computers?"](https://www.percepta.ai/blog/can-llms-be-computers) (Christos Tzamos, March 2026)
- Built on top of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- Public reimplementation only; exact Percepta internals are not public
