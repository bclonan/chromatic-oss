"""Train a selected representation with the same harness as the controls."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_benchmarks import main as benchmarks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["chroma", "chroma8", "fp32", "fp16", "bf16", "int8", "int4", "rgb24", "fixed24", "latent"], default="chroma")
    parser.add_argument("--task", choices=["scalar", "mnist", "cifar", "language"], default="scalar")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=["pilot", "full", "smoke"], default="pilot")
    args = parser.parse_args()
    representation = "chroma8" if args.representation == "chroma" else args.representation
    benchmarks(["--tasks", args.task, "--representations", representation, "--seeds", str(args.seed), "--profile", args.profile, "--output", str(Path("results") / f"train_{args.task}_{representation}.json")])


if __name__ == "__main__":
    main()
