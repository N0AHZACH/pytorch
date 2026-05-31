# Helper module for testing that infer_schema works with
# `from __future__ import annotations` even when `torch` is NOT imported
# at module level (only imported locally inside a function).
# See: https://github.com/pytorch/pytorch/issues/XXXXX
from __future__ import annotations


def get_fn_with_torch_tensor_annotation():
    """Return a function whose annotation is 'torch.Tensor' (stringified)
    but whose module globals do NOT contain 'torch'."""
    import torch

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    return fn
