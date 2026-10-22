import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tree(relative_path: str) -> ast.AST:
    return ast.parse((ROOT / relative_path).read_text())


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_subclass_datasets_forward_resize_multiple_to_their_base():
    # These datasets do not resize frames themselves; they must accept
    # resize_multiple and hand it to the base class that does. Each half of
    # this contract has been broken independently (parameter without the
    # forward, forward without the parameter), so pin both.
    subclass_paths = (
        "model_eval/datasets/dl3dv_hdf5_eval.py",
        "model_eval/datasets/dl3dv_reconstruction_eval.py",
    )

    for path in subclass_paths:
        inits = [
            node for node in ast.walk(_tree(path)) if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        assert inits, f"{path} has no __init__"
        for init in inits:
            arg_names = [arg.arg for arg in init.args.args + init.args.kwonlyargs]
            assert "resize_multiple" in arg_names, f"{path} __init__ does not accept resize_multiple"
            super_calls = [
                node for node in ast.walk(init) if isinstance(node, ast.Call) and _call_name(node) == "__init__"
            ]
            assert any(
                keyword.arg == "resize_multiple" for call in super_calls for keyword in call.keywords
            ), f"{path} __init__ does not forward resize_multiple to its base"


def test_run_inference_uses_shared_context_parallel_argument():
    calls = [node for node in ast.walk(_tree("model_eval/run_inference.py")) if isinstance(node, ast.Call)]
    context_parallel_declarations = [
        call
        for call in calls
        if _call_name(call) == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--context_parallel_size"
    ]
    assert not context_parallel_declarations
