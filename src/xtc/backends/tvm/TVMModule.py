from typing import Any
from typing_extensions import override
import xtc.itf as itf

from .TVMOps import TVMOperation
from .TVMEvaluator import TVMExecutor, TVMEvaluator

__all__ = [
    "TVMModule",
]

class TVMModule(itf.comp.Module):
    def __init__(
            self,
            operation: TVMOperation,
            name: str,
            payload_name: str,
            file_name: str,
            file_type: str,
            **kwargs: Any,
    ) -> None:
        self._operation = operation
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._file_type = file_type
        assert self._file_type == "shlib", "only support shlib for TVM Module"
        assert self._file_name.endswith(".so"), "file name is not a shlib"
        self._bare_ptr = kwargs.get("bare_ptr", False)

    @property
    @override
    def file_type(self) -> str:
        return self._file_type

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def payload_name(self) -> str:
        return self._payload_name

    @property
    @override
    def file_name(self) -> str:
        return self._file_name

    @override
    def export(self) -> None:
        pass

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        return TVMEvaluator(self, **kwargs)

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        return TVMExecutor(self, **kwargs)
