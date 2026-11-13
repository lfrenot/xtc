#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# Only the convolution sizes evaluated in the Ttile paper (Tollenaere et al.,
# "Autotuning Convolutions Is Easier Than You Think", TACO 2023, Fig. 8):
# Yolo9000 and ResNet18. MobilNet and the matmul sizes of prob_sizes are not
# registered.
from xtc.schedules.ttile.prob_sizes import (
    ddsizes_Yolo,
    ddsizes_RN18,
)

from .operations import register_operation

__all__: list[str] = []


def _register_conv2d_ops():
    map = dict(
        h="y",
        w="x",
        r="h",
        s="w",
    )
    for group in [ddsizes_Yolo, ddsizes_RN18]:
        for name, params in group.items():
            register_operation(
                "conv2d",
                name,
                {k: params[map.get(k, k)] for k in ["n", "h", "w", "f", "r", "s", "c"]},
                {"SH": params["stry"], "SW": params["strx"]},
            )


def _register_operations():
    _register_conv2d_ops()
