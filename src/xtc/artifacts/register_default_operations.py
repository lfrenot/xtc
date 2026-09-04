#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from .operations import register_operation

__all__: list[str] = []

_matmul_ops = dict(
    Vit_query={"i": 197, "j": 768, "k": 768},
    Vit_intermediate={"i": 197, "j": 3072, "k": 768},
    Vit_output={"i": 197, "j": 768, "k": 3072},
    Whisper_k_proj={"i": 1500, "j": 1280, "k": 1280},
    Whisper_fc1={"i": 1500, "j": 5120, "k": 1280},
    Whisper_fc2={"i": 1500, "j": 1280, "k": 5120},
    Llama31_8B_64_k_proj={"i": 64, "j": 1024, "k": 4096},
    Llama31_8B_64_gate_proj={"i": 64, "j": 14336, "k": 4096},
    Llama31_8B_64_down_proj={"i": 64, "j": 4096, "k": 14336},
)


def _register_matmul_ops():
    for name, params in _matmul_ops.items():
        register_operation(
            "matmul",
            name,
            {k: params[k] for k in ["i", "j", "k"]},
        )


def _register_operations():
    _register_matmul_ops()
