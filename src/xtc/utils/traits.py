#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Type


def add_traits(*traits: Type):
    """
    Add methods/attributes from trait classes to the decorated class.
    """

    def decorator(cls: Type):
        for trait in traits:
            for name, value in trait.__dict__.items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                if hasattr(cls, name):
                    raise AttributeError(
                        f"Trait {trait.__name__} conflict: {cls.__name__}.{name} already exists. "
                    )
                setattr(cls, name, value)
        return cls

    return decorator
