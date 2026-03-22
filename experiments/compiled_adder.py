"""Standalone demo: compiled binary adder using the Mentat framework.

Run: python experiments/compiled_adder.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import torch
from nanochat.mentat.base import CompiledBlock
from nanochat.mentat.adder import BinaryAdder

def execute_addition(block, adder, a, b, width, device):
    """Feed A/B tokens one pair at a time, block predicts R tokens."""
    A0, A1, BA0, BA1, R00 = adder.A0, adder.A1, adder.BA0, adder.BA1, adder.R00
    d = block.qkv.weight.shape[1]
    a_bits = [(a >> i) & 1 for i in range(width)]
    b_bits = [(b >> i) & 1 for i in range(width)]

    # Build token embeddings lookup
    emb_table = torch.zeros(16, d)
    for tok, vec in block.compiled_embeddings.items():
        emb_table[tok, :len(vec)] = vec

    tokens = [1, R00]  # BOS, initial carry
    with torch.no_grad():
        for i in range(width + 1):
            ai = a_bits[i] if i < width else 0
            bi = b_bits[i] if i < width else 0
            tokens.extend([A1 if ai else A0, BA1 if bi else BA0])
            x = emb_table[tokens].unsqueeze(0).to(device)
            out = block(x)
            # Read R-token logits from dims 8-9
            s_val = out[0, -1, 8].item()
            c_val = out[0, -1, 9].item()
            r_tok = adder.r_token(int(s_val > 0.5), int(c_val > 0.5))
            tokens.append(r_tok)

    result = 0
    for i in range(width + 1):
        if adder.r_sum(tokens[4 + 3 * i]):
            result |= 1 << i
    return result

def main():
    random.seed(42)
    device = torch.device("cpu")
    adder = BinaryAdder()
    block = CompiledBlock(adder, d_model=64).to(device)
    n = sum(p.numel() for p in block.parameters())

    print("=" * 60)
    print("  Mentat Compiled Adder Demo")
    print(f"  {n:,} frozen parameters, zero training")
    print("=" * 60)

    for bits in [8, 16, 32, 64]:
        ok = 0
        t0 = time.time()
        for _ in range(500):
            a = random.randint(0, 2**bits - 1)
            b = random.randint(0, 2**bits - 1)
            if execute_addition(block, adder, a, b, bits, device) == a + b:
                ok += 1
        print(f"  {bits:>2d}-bit: {ok}/500 = {ok/500:.1%}  ({time.time()-t0:.1f}s)")

    print("=" * 60)

if __name__ == "__main__":
    main()
