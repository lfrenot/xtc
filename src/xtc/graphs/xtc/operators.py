from typing_extensions import override
from typing import TypeAlias, cast
import numpy as np

from xtc.itf.operator import Operator
from xtc.itf.data import Tensor, TensorType

from .data import XTCTensor, XTCTensorType

__all__ = [
    "XTCOperator",
]


XTCOperPaddingAttr: TypeAlias = int|tuple[int]|tuple[int,int]|tuple[int,int,int,int]
XTCOperStrideAttr: TypeAlias = int|tuple[int]|tuple[int,int]


class XTCOperator(Operator):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        return inputs_types

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return inputs


class XTCOperTensor(XTCOperator):
    def __init__(self) -> None:
        super().__init__("tensor")


class XTCOperMatmul(XTCOperator):
    def __init__(self) -> None:
        super().__init__("matmul")

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # assume (IK, KJ) inputs and IJ output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) == 2
        assert len(inputs_types[1].shape) == 2
        i, k = cast(XTCTensorType, inputs_types[0]).constant_shape
        bk, j = cast(XTCTensorType, inputs_types[1]).constant_shape
        assert k == bk
        return [
            XTCTensorType(
                shape=(i, j),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        matmul = XTCTensor(np.matmul(inputs[0].numpy(), inputs[1].numpy()))
        return [matmul]

class XTCOperRelu(XTCOperator):
    def __init__(self, threshold: float = 0) -> None:
        super().__init__("relu")
        self._threshold = threshold

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        relu = XTCTensor(np.maximum(inputs[0].numpy(), self._threshold))
        return [relu]

class XTCOperConv2D(XTCOperator):
    def __init__(self, stride:  XTCOperStrideAttr = 1) -> None:
        super().__init__("conv2d")
        if isinstance(stride, int):
            self._sride: tuple[int,...] = tuple([stride]*2)
        else:
            assert isinstance(stride, tuple), f"padding for pad2d of wrong type, expect int or tuple: {stride}"
            if len(stride) == 1:
                self._padding = tuple([stride[0]]*4)
            else:
                assert len(stride) == 2, f"stride for conv2d of wrong size, expected 1 or 2: {stride}"
                self._stride = stride

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # TODO: assume (HWC, RSCF) inputs and HWF output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) >= 3
        assert len(inputs_types[1].shape) == 4
        inp_shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        weight_shape = cast(XTCTensorType, inputs_types[1]).constant_shape
        h, w, c = inp_shape[:-3]
        _, _, wc, f = weight_shape[:-4]
        assert c == wc
        sh, sw = self._stride
        oh, ow = (h - 1)//sh + 1, (w - 1)//sw + 1
        return [
            XTCTensorType(
                shape=tuple(
                    [
                        *inputs_types[0].shape[:-3],
                        oh,
                        ow,
                        f
                    ]
                ),
                dtype=inputs_types[0].dtype,
            ),
        ]


class XTCOperPad2D(XTCOperator):
    def __init__(self, padding: XTCOperPaddingAttr = 0) -> None:
        super().__init__("pad2d")
        if isinstance(padding, int):
            self._padding: tuple[int,...] = tuple([padding]*4)
        else:
            assert isinstance(padding, tuple), f"padding for pad2d of wrong type, expect int or tuple: {padding}"
            if len(padding) == 1:
                self._padding = tuple([padding[0]]*4)
            elif len(padding) == 2:
                self._padding = tuple([*([padding[0]]*2), *([padding[1]]*2)])
            else:
                assert len(padding) == 4, f"padding for pad2d of wrong size, expected 1, 2 or 4: {padding}"
                self._padding = padding

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # TODO: assume CHW input
        assert len(inputs_types) == 1
        assert inputs_types[0].shape is not None
        assert len(inputs_types[0].shape) >= 2
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        return [
            XTCTensorType(
                shape=tuple(
                    [
                        *inputs_types[0].shape[:-2],
                        shape[-2]+self._padding[0]+self._padding[1],
                        shape[-1]+self._padding[2]+self._padding[3],
                    ]
                ),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        inp_shape = cast(XTCTensorType, inputs[0].type).constant_shape
        pad_2d = [(self._padding[0], self._padding[1]), (self._padding[2], self._padding[3])]
        pads = [(0, 0) for _ in range(len(inp_shape)-2)] + pad_2d
        return [
            XTCTensor(data=np.pad(inputs[0].numpy(), pads)),
        ]
