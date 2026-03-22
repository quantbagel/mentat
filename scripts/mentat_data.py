"""Generate a Mentat synthetic corpus in nanochat's parquet format.

This writes parquet shards with a single `text` column, compatible with
`scripts.tok_train` and `scripts.base_train`.

Example:
    export NANOCHAT_DATA_DIR="$HOME/.cache/nanochat/base_data_mentat_stack"
    python -m scripts.mentat_data --train-docs 200000 --val-docs 5000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from nanochat.mentat.data import generate_trace_example


def default_output_dir() -> str:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanochat"))
    return os.environ.get("NANOCHAT_DATA_DIR", os.path.join(base_dir, "base_data_mentat_stack"))


def write_shard(output_dir: str, shard_index: int, docs: list[str], row_group_size: int) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - local environment may not have pyarrow
        raise SystemExit(
            "pyarrow is required for Mentat parquet generation. Install project deps first "
            "(for example `uv sync`) and rerun."
        ) from exc
    shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
    table = pa.Table.from_pydict({"text": docs})
    pq.write_table(
        table,
        shard_path,
        row_group_size=row_group_size,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    print(f"Wrote {shard_path} with {len(docs):,} documents")


def generate_split(
    output_dir: str,
    split: str,
    num_docs: int,
    shard_index: int,
    target_chars_per_shard: int,
    row_group_size: int,
    value_domain: int,
    max_stack_depth: int,
    min_program_len: int,
    max_program_len: int,
) -> int:
    docs: list[str] = []
    shard_chars = 0
    for index in range(num_docs):
        example = generate_trace_example(
            index=index,
            split=split,
            value_domain=value_domain,
            max_stack_depth=max_stack_depth,
            min_program_len=min_program_len,
            max_program_len=max_program_len,
        )
        docs.append(example.text)
        shard_chars += len(example.text)
        enough_chars = shard_chars >= target_chars_per_shard
        aligned = len(docs) % row_group_size == 0
        if enough_chars and aligned and split == "train":
            write_shard(output_dir, shard_index, docs, row_group_size)
            shard_index += 1
            docs = []
            shard_chars = 0

    if docs:
        write_shard(output_dir, shard_index, docs, row_group_size)
        shard_index += 1
    return shard_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mentat synthetic parquet shards")
    parser.add_argument("--output-dir", type=str, default=None, help="Parquet output directory")
    parser.add_argument("--train-docs", type=int, default=200_000, help="Number of training documents")
    parser.add_argument("--val-docs", type=int, default=5_000, help="Number of validation documents")
    parser.add_argument("--target-chars-per-shard", type=int, default=25_000_000, help="Approximate characters per train shard")
    parser.add_argument("--row-group-size", type=int, default=1024, help="Parquet row group size")
    parser.add_argument("--value-domain", type=int, default=4, help="Stack-machine arithmetic domain")
    parser.add_argument("--max-stack-depth", type=int, default=3, help="Maximum stack depth")
    parser.add_argument("--min-program-len", type=int, default=4, help="Minimum program length before HALT")
    parser.add_argument("--max-program-len", type=int, default=18, help="Maximum program length before HALT")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing")
    parser.add_argument("--preview", type=int, default=0, help="Print N sample documents and exit")
    args = parser.parse_args()

    if args.preview > 0:
        for i in range(args.preview):
            ex = generate_trace_example(
                index=i,
                split="train",
                value_domain=args.value_domain,
                max_stack_depth=args.max_stack_depth,
                min_program_len=args.min_program_len,
                max_program_len=args.max_program_len,
            )
            print("=" * 80)
            print(ex.text)
        return

    output_dir = args.output_dir or default_output_dir()

    if os.path.exists(output_dir):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it, or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    shard_index = 0
    shard_index = generate_split(
        output_dir=output_dir,
        split="train",
        num_docs=args.train_docs,
        shard_index=shard_index,
        target_chars_per_shard=args.target_chars_per_shard,
        row_group_size=args.row_group_size,
        value_domain=args.value_domain,
        max_stack_depth=args.max_stack_depth,
        min_program_len=args.min_program_len,
        max_program_len=args.max_program_len,
    )
    generate_split(
        output_dir=output_dir,
        split="val",
        num_docs=args.val_docs,
        shard_index=shard_index,
        target_chars_per_shard=args.target_chars_per_shard,
        row_group_size=args.row_group_size,
        value_domain=args.value_domain,
        max_stack_depth=args.max_stack_depth,
        min_program_len=args.min_program_len,
        max_program_len=args.max_program_len,
    )

    manifest = {
        "dataset": "mentat_stack",
        "train_docs": args.train_docs,
        "val_docs": args.val_docs,
        "target_chars_per_shard": args.target_chars_per_shard,
        "row_group_size": args.row_group_size,
        "value_domain": args.value_domain,
        "max_stack_depth": args.max_stack_depth,
        "min_program_len": args.min_program_len,
        "max_program_len": args.max_program_len,
        "output_dir": output_dir,
        "generated_at_unix": time.time(),
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
