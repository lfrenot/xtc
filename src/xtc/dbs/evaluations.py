#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import itertools
from pathlib import Path
import importlib.metadata
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, TypeAlias, TypeVar
import re

from xtc.utils.traits import add_traits
from xtc.itf.graph import Graph as Graph
from xtc.itf.comp import Module as Module
from xtc.itf.comp import Compiler as ICompiler
from xtc.itf.schd import Schedule as ISchedule
from xtc.itf.search.strategy import Strategy

from .utils import get_dict_digest

T = TypeVar("T")

Scalars: TypeAlias = str | int | float | bool
Structured: TypeAlias = list | dict | Scalars


class DataClassDict:
    def to_dict(self) -> dict[str, Any]:
        assert is_dataclass(self)
        return asdict(self)


class DataClassDigest:
    def get_digest(self) -> str:
        assert is_dataclass(self)
        return get_dict_digest(asdict(self))


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Platform:
    hostname: str
    system: str
    target: str


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Compiler:
    name: str
    version: str
    target: str
    threads: int
    backend: str


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Operation:
    name: str
    clsname: str
    clsargs: dict[str, Structured]
    payload: Structured

    def args_list(self, key: str) -> list[Any]:
        vals = self.clsargs[key]
        if isinstance(vals, dict):
            return list(vals.values())
        elif isinstance(vals, list):
            return vals
        return [vals]


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Schedule:
    clsname: str
    clsargs: dict[str, Structured]
    payload: Structured


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Payload:
    platform: Platform
    compiler: Compiler
    operation: Operation
    schedule: Schedule


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Result:
    metric: str
    values: list[float]


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Evaluation:
    payload: Payload
    code: int
    msg: str
    results: list[Result]


@add_traits(DataClassDict, DataClassDigest)
@dataclass(frozen=True)
class Tag:
    name: str
    payload: Payload


def get_compiler_xtc(
    backend: str, target: str = "native", threads: int = 1
) -> Compiler:
    xtc_version = importlib.metadata.version("xtc")
    return Compiler(
        name="xtc",
        version=xtc_version,
        target=target,
        threads=threads,
        backend=backend,
    )


def get_compiler(
    compiler: ICompiler,
) -> Compiler:
    cls = compiler.__class__
    backend = cls.__module__.split(".")[-1]
    return get_compiler_xtc(
        backend=backend,
        target="native",  ##TODO
        threads=1,  # TODO
    )


def get_platform(
    hostname: str = "", system: str = "native", target: str = "native"
) -> Platform:
    import platform

    if hostname == "":
        assert system == "native" and target == "native"
        hostname = platform.node().split(".")[0]
    if system == "native" or target == "native":
        assert system == "native" and target == "native"
        system = platform.system()
        target = platform.machine()
    return Platform(
        hostname=hostname.lower(),
        system=system.lower(),
        target=target.lower(),
    )


def get_operation_graph(
    graph: Graph,
    graph_args: dict[str, Structured] = {},
) -> Operation:
    cls = graph.__class__
    graph_str = str(graph)
    # TODO: As of now, need to rename ssa variables
    # May provide a node.uid independent dump for graph
    var_map = {
        re.escape(m.group(0)): f"%{idx}"
        for idx, m in enumerate(re.finditer(r"%\d+", graph_str))
    }
    graph_str = re.sub(
        "|".join([k for k in var_map.keys()]),
        lambda m: var_map[m.group(0)],
        graph_str,
    )
    return Operation(
        clsname=f"{cls.__module__}.{cls.__qualname__}",
        clsargs={**graph_args},
        name=graph.name,
        payload=graph_str,
    )


def get_operation_from_artifacts(
    operator: str,
    name: str,
    dtype: str,
) -> Operation:
    from xtc.artifacts.operations import get_operation

    op = get_operation(operator, name)
    return get_operation_from_dims(operator, op["dims"], op["params"], dtype)


def get_operation_from_dims(
    operator: str,
    dims: dict,
    params: dict,
    dtype: str,
) -> Operation:
    return Operation(
        name=operator,
        clsname=f"xtc.artifacts.factory.create_graph",
        clsargs=dict(operator=operator, dims=dims, params=params, dtype=dtype),
        payload="",
    )


def get_schedule_strategy(
    strategy: Strategy,
    strategy_args: dict[str, Structured],
    schedule: Any,
) -> Schedule:
    cls = strategy.__class__
    return Schedule(
        clsname=f"{cls.__module__}.{cls.__qualname__}",
        clsargs={**strategy_args},
        payload=str(schedule),
    )


def get_schedule(
    schedule: ISchedule,
    schedule_args: dict[str, Structured] = {},
) -> Schedule:
    cls = schedule.__class__
    return Schedule(
        clsname=f"{cls.__module__}.{cls.__qualname__}",
        clsargs={**schedule_args},
        payload=str(schedule),
    )


def get_result(
    metric: str,
    values: list[float],
) -> Result:
    return Result(
        metric=metric,
        values=values,
    )


def get_payload(
    platform: Platform,
    compiler: Compiler,
    operation: Operation,
    schedule: Schedule,
) -> Payload:
    return Payload(
        platform=platform,
        compiler=compiler,
        operation=operation,
        schedule=schedule,
    )


def get_evaluation(
    payload: Payload,
    results: list[Result],
    code: int,
    msg: str = "",
) -> Evaluation:
    return Evaluation(
        payload=payload,
        results=results,
        code=code,
        msg=msg,
    )


def get_tag(
    name: str,
    payload: Structured = "",
) -> Tag:
    return Tag(
        name=name,
        payload=payload,  # type: ignore
        # TODO: Christophe
    )


class EvaluationsDB:
    def __init__(self, db_path: Path | str, force_create: bool = False):
        from xtc.dbs.backends.sqlite import db_evaluations

        self._db = db_evaluations.EvaluationsORM(
            db_path,
            allow_migration=False,
            force_create=force_create,
        )

    def create_unique_tag(self, tag: Tag):
        self._db.create_unique_tag(tag)

    def get_or_create_tag(self, tag: Tag):
        self._db.get_or_create_tag(tag)

    def delete_filtered_tags(self, **kwargs: Any) -> list[int]:
        return self._db.delete_filtered_tags(**kwargs)

    def delete_filtered_evaluations(self, **kwargs: Any) -> list[int]:
        return self._db.delete_filtered_evaluations(**kwargs)

    def tag_evaluation(self, evaluation_id: int, tags: list[str]):
        self._db.tag_evaluation(evaluation_id, tags)

    def record_evaluation(self, evaluation: Evaluation, tags: list[str] = []):
        self._db.record_evaluation(evaluation, tags)

    def record_evaluations(self, evaluations: list[Evaluation], tags: list[str] = []):
        self._db.record_evaluations(evaluations, tags)

    def get_payload_evaluations(self, payload: Payload) -> dict[int, Evaluation]:
        return self._db.get_payload_evaluations(payload)

    def get_operation_evaluations(self, operation: Operation) -> dict[int, Evaluation]:
        return self._db.get_operation_evaluations(operation)

    def get_last_payload_metric_evaluation(
        self, payload: Payload, metric: str = "elapsed"
    ) -> tuple[None | int, Evaluation]:
        evaluations = self._db.get_payload_evaluations(payload)
        for id, evaluation in reversed(evaluations.items()):
            if (
                len(evaluation.results) == 1
                and evaluation.results[0].metric == metric
                and len(evaluation.results[0].values) > 0
            ):
                return (id, evaluation)
        return (None, None)  # type: ignore
        # TODO: Christophe

    def get_filtered_metric_evaluations(
        self,
        metric: str = "elapsed",
        full: bool = True,
        raw: bool = False,
        **kwargs: Any,
    ) -> dict[int, Evaluation]:
        return self._db.get_filtered_metric_evaluations(
            metric=metric, full=full, raw=raw, **kwargs
        )

    def get_filtered_evaluations(
        self, full: bool = True, raw: bool = False, **kwargs: Any
    ) -> dict[int, Evaluation]:
        return self._db.get_filtered_evaluations(full=full, raw=raw, **kwargs)

    def get_from_digest(self, target_cls: str, digest: str) -> Any:
        return self._db.get_from_digest(target_cls, digest)

    def dump_all(self, format: str = "jsonl", verbose: bool = False):
        self._db.dump_all(format=format, verbose=verbose)

    def dump_filtered(
        self, format: str = "jsonl", verbose: bool = False, **kwargs: Any
    ):
        self._db.dump_filtered(format=format, verbose=verbose, **kwargs)

    def dump_tags(self, format: str = "jsonl", verbose: bool = False):
        self._db.dump_tags(format=format, verbose=verbose)

    def get_filtered_operations(self, **kwargs: Any) -> dict[int, Operation]:
        return self._db.get_filtered_operations(**kwargs)

    def get_filtered_tags(self, **kwargs: Any) -> dict[int, Tag]:
        return self._db.get_filtered_tags(**kwargs)


class DBEvaluator:
    def __init__(
        self,
        db: EvaluationsDB,
        module: Module,
        payload: Payload = None,  # type: ignore
        cached: bool = False,
        **kwargs: Any,
    ):
        self.db = db
        self.payload = payload
        if self.payload is None:
            self.compiler = get_compiler(kwargs["compiler"])
            self.platform = get_platform()
            self.operation = get_operation_graph(kwargs["graph"])
            self.schedule = get_schedule(kwargs["schedule"])
            self.payload = Payload(
                platform=self.platform,
                compiler=self.compiler,
                operation=self.operation,
                schedule=self.schedule,
            )
        self.module = module
        self.cached = cached
        self._evaluator = module.get_evaluator(**kwargs)
        self._repeat = getattr(self._evaluator, "_repeat", 1)
        self._metrics = getattr(self._evaluator, "_pmu_counters", [])
        if not self._metrics:
            self._metrics = ["elapsed"]

    def get_results(self) -> list[float]:
        if not self.cached:
            return []
        evaluations = self.db.get_payload_evaluations(self.payload)
        results_map: dict[Any, list] = {k: [] for k in self._metrics}
        for evaluation in reversed(evaluations.values()):
            if evaluation.code != 0:
                continue
            for result in reversed(evaluation.results):
                if (
                    result.metric in results_map
                    and len(results_map[result.metric]) == 0
                    and len(result.values) == self._repeat
                ):
                    results_map[result.metric] = result.values
        if not all([len(v) == self._repeat for v in results_map.values()]):
            return []
        values = list(itertools.chain(*zip(*[v for v in results_map.values()])))
        return values

    def evaluate(self, **kwargs: Any) -> tuple[list[float], int, str]:
        values = self.get_results()
        if values:
            return values, 0, ""
        values, code, msg = self._evaluator.evaluate(**kwargs)
        evaluation = Evaluation(
            payload=self.payload,
            code=code,
            msg=msg,
            results=[
                Result(
                    metric=metric,
                    values=values[
                        i * len(self._metrics) : (i + 1) * len(self._metrics)
                    ],
                )
                for i, metric in enumerate(self._metrics)
            ],
        )
        self.db.record_evaluation(evaluation)
        return values, code, msg
