"""Binary full-adder compiled into transformer weights.

This is the reference implementation of a CompiledModule. It compiles the
8-row full-adder truth table into:
  - 3 type-selective 2D attention heads (select A, B, R tokens)
  - 8 ReLU indicator units in the FFN (one per truth-table row)
  - Output routing to sum/carry dimensions

Works at ANY bit width with 100% accuracy. Zero training required.

Token types:
  A0/A1   — operand A bit (0 or 1)
  BA0/BA1 — operand B bit (0 or 1)
  R00/R10/R01/R11 — result (sum_bit, carry_out)

Execution trace format:
  ... = R00 A_0 BA_0 [R_0] A_1 BA_1 [R_1] ... [R_n] EOS

Each R token is computed by the compiled block from the 3 preceding tokens:
  position p   (self):  BA_i  → b_i value
  position p-1:         A_i   → a_i value
  position p-2:         R_{i-1} → carry_in
"""

import torch
from nanochat.mentat.base import CompiledModule, CompiledWeights


class BinaryAdder(CompiledModule):
    """Full-adder compiled into 2D attention heads + ReLU truth table."""

    compiled_dims = 10
    window_size = 3
    n_active_heads = 3
    ffn_hidden = 16

    # Token IDs (these must match the host model's vocabulary)
    # Override these when integrating with a specific tokenizer
    A0 = 8;  A1 = 9
    BA0 = 10; BA1 = 11
    R00 = 12; R10 = 13; R01 = 14; R11 = 15

    TRUTH_TABLE = [
        # a  b  c  sum carry
        (0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 1, 0, 1),
        (1, 0, 0, 1, 0),
        (1, 0, 1, 0, 1),
        (1, 1, 0, 0, 1),
        (1, 1, 1, 1, 1),
    ]

    @staticmethod
    def r_token(s: int, c: int) -> int:
        return [12, 13, 14, 15][s + 2 * c]  # R00, R10, R01, R11

    @staticmethod
    def r_sum(tok: int) -> int:
        return 1 if tok in (13, 15) else 0  # R10 or R11

    def compile(self, d_model: int, n_heads: int) -> CompiledWeights:
        S = 10.0
        d = d_model

        # ── Embeddings (compiled dims 0-9 only) ──
        embeddings = {}
        for tok, vec in [
            (self.A0,  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            (self.A1,  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            (self.BA0, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            (self.BA1, [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            (self.R00, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            (self.R10, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            (self.R01, [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
            (self.R11, [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
        ]:
            embeddings[tok] = torch.tensor(vec, dtype=torch.float32)

        # ── QKV weights ──
        w = torch.zeros(3 * d, d)
        # Q: at BA-type positions (dim 3 = type_B flag)
        w[1, 3] = S                    # head 0: select BA → Q=[0, S]
        w[2, 3] = S                    # head 1: select A  → Q=[S, 0]
        w[4, 3] = -S; w[5, 3] = -S    # head 2: select R  → Q=[-S, -S]

        # K: type encoding (shared by all heads)
        for h in range(3):
            row = d + 2 * h
            w[row, 2] = S;     w[row, 4] = -S      # K[0] = S·typeA - S·typeR
            w[row + 1, 3] = S; w[row + 1, 4] = -S  # K[1] = S·typeB - S·typeR

        # V: payload extraction
        for h in range(2):
            w[2 * d + 2 * h, 0] = 1.0  # heads 0,1: bit value
        w[2 * d + 4, 1] = 1.0          # head 2: carry

        # ── Out projection → dims 5, 6, 7 ──
        op = torch.zeros(d, d)
        op[5, 0] = 1.0   # b_val → dim 5
        op[6, 2] = 1.0   # a_val → dim 6
        op[7, 4] = 1.0   # carry → dim 7

        # ── FFN up: 8 indicator units ──
        w1 = torch.zeros(16, d)
        b1 = torch.zeros(16)
        for i, (a0, b0, c0, _, _) in enumerate(self.TRUTH_TABLE):
            w1[i, 6] = S * 2 * (2 * a0 - 1)
            w1[i, 5] = S * 2 * (2 * b0 - 1)
            w1[i, 7] = S * 2 * (2 * c0 - 1)
            b1[i] = S * (1 - 2 * (a0 + b0 + c0))

        # ── FFN down: sum → dim 8, carry → dim 9 ──
        w2 = torch.zeros(d, 16)
        b2 = torch.zeros(d)
        for i, (_, _, _, s, co) in enumerate(self.TRUTH_TABLE):
            w2[8, i] = float(s)
            w2[9, i] = float(co)

        # ── Output head rows for R tokens ──
        head_weights = {
            self.R00: (self._head_row(d, -1, -1), S),
            self.R10: (self._head_row(d, +1, -1), 0.0),
            self.R01: (self._head_row(d, -1, +1), 0.0),
            self.R11: (self._head_row(d, +1, +1), -S),
        }

        return CompiledWeights(
            qkv=w, out_proj=op,
            ffn_up_w=w1, ffn_up_b=b1,
            ffn_down_w=w2, ffn_down_b=b2,
            embeddings=embeddings,
            head_weights=head_weights,
        )

    @staticmethod
    def _head_row(d: int, sum_sign: int, carry_sign: int) -> torch.Tensor:
        row = torch.zeros(d)
        row[8] = float(sum_sign)
        row[9] = float(carry_sign)
        return row

    def format_trace(self, a: int, b: int, width: int) -> list[int]:
        """Generate execution trace tokens for a+b at given width."""
        trace = [self.R00]  # initial carry = 0
        carry = 0
        for i in range(width + 1):
            ai = (a >> i) & 1 if i < width else 0
            bi = (b >> i) & 1 if i < width else 0
            total = ai + bi + carry
            s, carry = total % 2, total // 2
            trace.extend([
                self.A1 if ai else self.A0,
                self.BA1 if bi else self.BA0,
                self.r_token(s, carry),
            ])
        return trace
