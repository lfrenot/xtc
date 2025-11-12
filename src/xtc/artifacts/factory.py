#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph


def create_graph(operator: str, dims: dict, params: dict, dtype: str) -> XTCGraph:
    if operator == "matmul":
        return create_matmul_graph(dims, params, dtype)
    elif operator == "conv2d":
        return create_conv2d_graph(dims, params, dtype)
    else:
        assert False, f"operator {operator} not supported"


def create_matmul_graph(dims: dict, params: dict, dtype: str) -> XTCGraph:
    assert not params
    I, J, K = [dims[k] for k in ["i", "j", "k"]]
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")

    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="O")
    graph = gb.graph
    return graph


def create_conv2d_graph(dims: dict, params: dict, dtype: str) -> XTCGraph:
    N, H, W, F, R, S, C = [dims[k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
    SH, SW = [params[k] for k in ["SH", "SW"]]

    a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
    b = O.tensor((R, S, C, F), dtype)

    with O.graph(name="conv2d") as gb:
        O.conv2d(a, b, stride=(SH, SW), name="O")
    graph = gb.graph
    return graph
