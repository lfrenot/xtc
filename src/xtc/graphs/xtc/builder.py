from typing import Any

from .graph import XTCGraph
from .context import XTCGraphContext

class graph_builder():
    def __init__(self, **graph_kwargs: Any) -> None:
        self._graph_kwargs = graph_kwargs
        self._graph: XTCGraph|None = None

    def __enter__(self) -> "graph_builder":
        XTCGraphContext.push(**self._graph_kwargs)
        return self

    def __exit__(self, *_: Any) -> None:
        scope = XTCGraphContext.pop()
        self._graph = scope.graph

    @property
    def graph(self) -> XTCGraph:
        assert self._graph is not None, "can't get graph inside builder context"
        return self._graph
