"""
Run the full pipeline by executing every notebook in the correct order.
No shortcuts: real notebooks, full tuning, 50k synthetic samples.

Usage:
  python run_full_pipeline.py           # Run all 8 notebooks (01, 02, 06_wrought, 06_cast, 03, 04, 05_wrought, 05_cast)
  python run_full_pipeline.py --core-only   # Run only core 4 (01, 02, 06_wrought, 06_cast)

Requires: pip install nbconvert
"""
import argparse
import os
import sys
import time

# Avoid Windows ProactorEventLoop + zmq warning when nbconvert runs notebook kernels
if sys.platform == "win32":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

# Run from project directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def log(msg: str, flush: bool = True) -> None:
    """Print to terminal and flush so output appears immediately."""
    print(msg, flush=flush)


# Short description of each notebook (for terminal logs)
NOTEBOOK_DESCRIPTIONS = {
    "01_hyperparameter_tuning_forward.ipynb": "Forward hyperparameter tuning (per-target best model + params for wrought and cast)",
    "02_hyperparameter_tuning_backward.ipynb": "Backward GMM tuning (BIC) for wrought and cast composition space",
    "06_generate_synthetic_wrought.ipynb": "Generate synthetic wrought alloy pool (GMM sample + forward predictions, 50k rows)",
    "06_generate_synthetic_cast.ipynb": "Generate synthetic cast alloy pool (GMM sample + forward predictions, 50k rows)",
    "03_forward_wrought_alloys.ipynb": "Forward prediction: wrought alloys (train per-target models, predict properties)",
    "04_forward_cast_alloys.ipynb": "Forward prediction: cast alloys (train per-target models, predict properties)",
    "05_backward_wrought.ipynb": "Backward search: find wrought alloys matching target properties from synthetic pool",
    "05_backward_cast.ipynb": "Backward search: find cast alloys matching target properties from synthetic pool",
}

# Core pipeline (required for synthetic pools and backward search)
CORE_NOTEBOOKS = [
    "01_hyperparameter_tuning_forward.ipynb",
    "02_hyperparameter_tuning_backward.ipynb",
    "06_generate_synthetic_wrought.ipynb",
    "06_generate_synthetic_cast.ipynb",
]

# Optional: forward prediction and backward search (use default TARGETS in notebooks)
OPTIONAL_NOTEBOOKS = [
    "03_forward_wrought_alloys.ipynb",
    "04_forward_cast_alloys.ipynb",
    "05_backward_wrought.ipynb",
    "05_backward_cast.ipynb",
]

# Timeouts in seconds (01 and 06 are slow)
TIMEOUTS = {
    "01_hyperparameter_tuning_forward.ipynb": 900,
    "02_hyperparameter_tuning_backward.ipynb": 300,
    "06_generate_synthetic_wrought.ipynb": 600,
    "06_generate_synthetic_cast.ipynb": 600,
    "03_forward_wrought_alloys.ipynb": 300,
    "04_forward_cast_alloys.ipynb": 120,
    "05_backward_wrought.ipynb": 60,
    "05_backward_cast.ipynb": 60,
}
DEFAULT_TIMEOUT = 120


def run_notebook(path: str, timeout: int) -> None:
    """Execute a single notebook in place using nbconvert."""
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        print(
            "ERROR: nbconvert is required. Install with: pip install nbconvert",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout)
    ep.preprocess(nb, {"metadata": {"path": SCRIPT_DIR}})
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline (all notebooks in order).")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only core 4 notebooks (01, 02, 06_wrought, 06_cast).",
    )
    args = parser.parse_args()

    if args.core_only:
        notebooks = CORE_NOTEBOOKS
    else:
        notebooks = CORE_NOTEBOOKS + OPTIONAL_NOTEBOOKS

    total = len(notebooks)
    pipeline_start = time.time()

    log("")
    log("=" * 70)
    log("FULL PIPELINE – notebooks will run in this order:")
    log("=" * 70)
    for i, name in enumerate(notebooks, 1):
        desc = NOTEBOOK_DESCRIPTIONS.get(name, "Execute notebook")
        log(f"  {i}. {name}")
        log(f"     -> {desc}")
    log("=" * 70)
    log("")

    for i, name in enumerate(notebooks, 1):
        path = os.path.join(SCRIPT_DIR, name)
        if not os.path.isfile(path):
            print(f"ERROR: Notebook not found: {path}", file=sys.stderr)
            sys.exit(1)
        timeout = TIMEOUTS.get(name, DEFAULT_TIMEOUT)
        desc = NOTEBOOK_DESCRIPTIONS.get(name, "Execute notebook")

        log("")
        log("-" * 70)
        log(f"STEP {i}/{total} – NOW RUNNING: {name}")
        log(f"  What: {desc}")
        log(f"  Timeout: {timeout}s (this step will stop if it runs longer)")
        log("-" * 70)
        step_start = time.time()
        try:
            run_notebook(path, timeout)
            elapsed = time.time() - step_start
            log(f"  OK: completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - step_start
            log(f"  FAIL after {elapsed:.1f}s: {name}", flush=True)
            print(f"{e}", file=sys.stderr)
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    log("")
    log("=" * 70)
    log(f"Done. All {total} notebook(s) completed successfully in {total_elapsed:.1f}s total.")
    log("=" * 70)


if __name__ == "__main__":
    main()
