"""Regression test for the positional-flag swap in the KV-cache pipeline.

A positional call into ``generate_samples_from_batch`` once bound
``show_progress`` to ``ignore_neighbors``, silently dropping the reference
views on any process with ``show_progress=True`` (rank 0 -- every single-GPU
run). Guard against reintroduction by asserting the trailing flags are
keyword-only at the signature level.

The check is AST-based so it runs without torch/GPU dependencies.
"""

import ast
import unittest
from pathlib import Path

PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_training"
    / "pipeline"
    / "kv_cache_pipeline.py"
)

KEYWORD_ONLY_FLAGS = {
    "generate_samples_from_batch": (
        "ignore_neighbors",
        "show_progress",
        "progress_bar_leave",
    ),
    "denoise_to_latents": (
        "show_progress",
        "progress_bar_leave",
        "max_neighbors_per_encode",
    ),
    "forward_inference": (
        "show_progress",
        "progress_bar_leave",
        "max_neighbors_per_encode",
    ),
}


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {PIPELINE_PATH}")


class TestPipelineFlagsKeywordOnly(unittest.TestCase):
    def test_trailing_flags_are_keyword_only(self):
        tree = ast.parse(PIPELINE_PATH.read_text())
        for func_name, flags in KEYWORD_ONLY_FLAGS.items():
            func = _find_function(tree, func_name)
            kwonly = {arg.arg for arg in func.args.kwonlyargs}
            for name in flags:
                self.assertIn(
                    name,
                    kwonly,
                    f"{func_name}: {name} must be keyword-only: a positional "
                    "call once bound show_progress to ignore_neighbors, "
                    "dropping the reference views.",
                )


if __name__ == "__main__":
    unittest.main()
