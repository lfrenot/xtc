#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import logging
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
import sys
import shutil
from typing import Any, TextIO, Type, TypeVar

from sqlalchemy import (
    create_engine,
    select,
    update,
    DateTime,
    String,
    func,
    distinct,
    Engine,
    ForeignKey,
    and_,
)
from sqlalchemy.orm import (
    aliased,
    declarative_base,
    relationship,
    Session,
    mapped_column,
    Mapped,
    selectinload,
)

from xtc.utils.dump import dump_plain
from xtc.utils.traits import add_traits

from ...evaluations import Compiler as DCompiler
from ...evaluations import Platform as DPlatform
from ...evaluations import Operation as DOperation
from ...evaluations import Schedule as DSchedule
from ...evaluations import Result as DResult
from ...evaluations import Payload as DPayload
from ...evaluations import Evaluation as DEvaluation
from ...evaluations import Tag as DTag

from ...utils import (
    save_blob_dict,
    load_blob_dict,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationsORM",
]


T = TypeVar("T")

Base = declarative_base()


class ORMDict:
    def to_dict(self) -> dict[str, Any]:
        def deserialize(obj: Any):
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [deserialize(elt) for elt in obj]
            return deserialize_iso(obj)

        return {k: deserialize(v) for k, v in self.__dict__.items() if k[:1] != "_"}


class ORMDigestMixin:
    @classmethod
    def get_or_create_digest(
        cls: Type[T],
        session: Session,
        digest: str,
        **kwargs: Any,
    ) -> T:
        digest_field = getattr(cls, "digest", None)
        assert digest_field is not None
        stmt = select(cls).where(digest_field == digest)
        obj = session.execute(stmt).scalar_one_or_none()
        if obj is not None:
            return obj
        obj = cls(digest=digest, **kwargs)  # type: ignore
        session.add(obj)
        try:
            session.commit()
            return obj
        except IntegrityError as e:
            session.rollback()
            return session.execute(stmt).scalar_one()


class ORMUniqueMixin:
    @classmethod
    def get_or_create_unique(cls: Type[T], session: Session, **kwargs: Any) -> T:
        conditions = [getattr(cls, k) == v for k, v in kwargs.items()]
        stmt = select(cls).where(*conditions)
        obj = session.execute(stmt).scalar_one_or_none()
        if obj is not None:
            return obj
        obj = cls(**kwargs)
        session.add(obj)
        try:
            session.commit()
            return obj
        except IntegrityError:
            session.rollback()
            return session.execute(stmt).scalar_one()


class Version(Base):
    __tablename__ = "version"
    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[int]
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )


class Platform(Base, ORMDigestMixin):
    __tablename__ = "platforms"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    hostname: Mapped[str]
    system: Mapped[str]
    target: Mapped[str]
    digest: Mapped[str] = mapped_column(String, unique=True, index=True)

    @classmethod
    def from_plain(cls, session: Session, platform: DPlatform) -> "Platform":
        digest = serialize_data(session, platform)
        return Platform.get_or_create_digest(
            session,
            digest,
            hostname=platform.hostname,
            system=platform.system,
            target=platform.target,
        )

    def as_plain(self, session: Session, full: bool = False) -> DPlatform:
        return DPlatform(
            hostname=self.hostname,
            system=self.system,
            target=self.target,
        )


class Compiler(Base, ORMDigestMixin):
    __tablename__ = "compilers"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    name: Mapped[str]
    version: Mapped[str]
    target: Mapped[str]
    threads: Mapped[int]
    backend: Mapped[str]
    digest: Mapped[str] = mapped_column(String, unique=True, index=True)

    @classmethod
    def from_plain(cls, session: Session, compiler: DCompiler) -> "Compiler":
        digest = serialize_data(session, compiler)
        return Compiler.get_or_create_digest(
            session,
            digest,
            name=compiler.name,
            version=compiler.version,
            target=compiler.target,
            threads=compiler.threads,
            backend=compiler.backend,
        )

    def as_plain(self, session: Session, full: bool = False) -> DCompiler:
        return DCompiler(
            name=self.name,
            version=self.version,
            target=self.target,
            threads=self.threads,
            backend=self.backend,
        )


class Operation(Base, ORMDigestMixin):
    __tablename__ = "operations"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    name: Mapped[str]
    clsname: Mapped[str]
    clsargs: Mapped[str]
    digest: Mapped[str] = mapped_column(String, unique=True, index=True)

    @classmethod
    def from_plain(cls, session: Session, operation: DOperation) -> "Operation":
        digest = serialize_data(session, operation)
        return Operation.get_or_create_digest(
            session,
            digest,
            name=operation.name,
            clsname=operation.clsname,
            clsargs=str(operation.clsargs),
        )

    def as_plain(self, session: Session, full: bool = False) -> DOperation:
        if full:
            return deserialize_data(session, DOperation, self.digest)
        else:
            return DOperation(
                name=self.name, clsname=self.clsname, clsargs=self.clsargs, payload=""
            )


class Schedule(Base, ORMDigestMixin):
    __tablename__ = "schedules"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    clsname: Mapped[str]
    clsargs: Mapped[str]
    digest: Mapped[str] = mapped_column(String, unique=True, index=True)

    @classmethod
    def from_plain(cls, session: Session, schedule: DSchedule) -> "Schedule":
        digest = serialize_data(session, schedule)
        return Schedule.get_or_create_digest(
            session,
            digest,
            clsname=schedule.clsname,
            clsargs=str(schedule.clsargs),
        )

    def as_plain(self, session: Session, full: bool = False) -> DSchedule:
        if full:
            return deserialize_data(session, DSchedule, self.digest)
        else:
            return DSchedule(clsname=self.clsname, clsargs=self.clsargs, payload="")


class Payload(Base, ORMUniqueMixin):
    __tablename__ = "payloads"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    platform_id: Mapped[int] = mapped_column(
        ForeignKey("platforms.id"), nullable=False, index=True
    )
    platform: Mapped["Platform"] = relationship(backref="evaluations")
    compiler_id: Mapped[int] = mapped_column(
        ForeignKey("compilers.id"), nullable=False, index=True
    )
    compiler: Mapped["Compiler"] = relationship(backref="compilers")
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("operations.id"), nullable=False, index=True
    )
    operation: Mapped["Operation"] = relationship(backref="operations")
    schedule_id: Mapped[str] = mapped_column(
        ForeignKey("schedules.id"), nullable=False, index=True
    )
    schedule: Mapped["Schedule"] = relationship(backref="schedules")

    @classmethod
    def from_plain(cls, session: Session, payload: DPayload) -> "Payload":
        return Payload.get_or_create_unique(
            session,
            platform=Platform.from_plain(session, payload.platform),
            compiler=Compiler.from_plain(session, payload.compiler),
            operation=Operation.from_plain(session, payload.operation),
            schedule=Schedule.from_plain(session, payload.schedule),
        )

    def as_plain(self, session: Session, full: bool = False) -> DPayload:
        return DPayload(
            platform=self.platform.as_plain(session, full=full),
            compiler=self.compiler.as_plain(session, full=full),
            operation=self.operation.as_plain(session, full=full),
            schedule=self.schedule.as_plain(session, full=full),
        )


class EvaluationTag(Base, ORMUniqueMixin):
    __tablename__ = "evaluations_tags"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    evaluation_id: Mapped[int] = mapped_column(
        ForeignKey("evaluations.id"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id"), primary_key=True)


@add_traits(ORMDict)
class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    payload_id: Mapped[int] = mapped_column(
        ForeignKey("payloads.id"), nullable=False, index=True
    )
    payload: Mapped["Payload"] = relationship(backref="payloads")
    code: Mapped[int]
    msg: Mapped[str]
    results: Mapped[list["Result"]] = relationship(
        "Result",
        backref="evaluation",
        cascade="all, delete-orphan",
        order_by="Result.id",
    )
    tag_links: Mapped[list[EvaluationTag]] = relationship(
        backref="evaluation",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_plain(cls, session: Session, evaluation: DEvaluation) -> "Evaluation":
        return Evaluation(
            payload=Payload.from_plain(session, evaluation.payload),
            code=evaluation.code,
            msg=evaluation.msg,
            results=Result.from_plain(session, evaluation.results),
        )

    def as_plain(self, session: Session, full: bool = False) -> DEvaluation:
        return DEvaluation(
            payload=self.payload.as_plain(session, full=full),
            code=self.code,
            msg=self.msg,
            results=[r.as_plain(session, full=full) for r in self.results],
        )


@add_traits(ORMDict)
class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    metric: Mapped[str] = mapped_column(String)
    values: Mapped[list["ResultValue"]] = relationship(
        "ResultValue",
        backref="result",
        cascade="all, delete-orphan",
        order_by="ResultValue.id",
    )
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluations.id"))

    @classmethod
    def from_plain(cls, session: Session, results: list[DResult]) -> list["Result"]:
        return [
            Result(
                metric=result.metric,
                values=[ResultValue(value=value) for value in result.values],
            )
            for result in results
        ]

    def as_plain(self, session: Session, full: bool = False) -> DResult:
        return DResult(
            metric=self.metric,
            values=[v.value for v in self.values],
        )


@add_traits(ORMDict)
class ResultValue(Base):
    __tablename__ = "result_values"
    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[float]
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id"))


class Tag(Base, ORMDigestMixin):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    digest: Mapped[str] = mapped_column(String, unique=True, index=True)
    evaluation_links: Mapped[list[EvaluationTag]] = relationship(
        backref="tag",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_plain(cls, session: Session, tag: DTag) -> "Tag":
        digest = serialize_data(session, tag)
        return Tag.get_or_create_digest(
            session,
            name=tag.name,
            digest=digest,
        )

    def as_plain(self, session: Session, full: bool = False) -> DTag:
        if full:
            return deserialize_data(session, DTag, self.digest)
        else:
            return DTag(
                name=self.name,
                payload="",
            )


def serialize_data(session: Session, data: Any) -> str:
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    return save_blob_dict(
        _engine_blob_dir(engine),
        data.to_dict(),
    )


def deserialize_data(session: Session, cls: Type[T], digest: str) -> T:
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    params = load_blob_dict(
        _engine_blob_dir(engine),
        digest,
    )
    return cls(**params)


def deserialize_iso(obj: "Any"):
    if obj is None:
        return None
    if hasattr(obj, "tzinfo"):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        obj = obj.astimezone(local_tz)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _engine_blob_dir(engine: Engine) -> Path:
    db_path = Path(str(engine.url.database))
    return db_path.parent / "blobs"


class EvaluationsORM:
    CURRENT_VERSION: int = 1002000
    MIN_VERSION: int = 1002000

    _class_map = dict(
        platform=Platform,
        compiler=Compiler,
        operation=Operation,
        schedule=Schedule,
        tag=Tag,
    )

    def __init__(
        self,
        db_path: str | Path,
        allow_migration: bool = False,
        force_create: bool = False,
    ):
        self._db_path = Path(db_path)
        self._db_name = "evaluations"
        self._db_file = self._db_path / f"{self._db_name}.db"
        if self._db_exists() and not force_create:
            self._engine = self._open_db_with_migrations(allow_migration)
        else:
            self._db_remove()
            self._engine = self._db_create_engine(create=True)
            Base.metadata.create_all(self._engine)

    def _db_exists(self) -> bool:
        return self._db_file.exists()

    def _db_create_engine(self, create: bool = True) -> Engine:
        assert create == (not self._db_file.exists())
        self._db_path.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{self._db_file}")
        return engine

    def _db_remove(self):
        # Note: we remove the whole db dir there
        shutil.rmtree(self._db_path, ignore_errors=True)

    @classmethod
    def _get_actual_version(cls, session: Session) -> int:
        row = session.get(Version, 1)
        if row is None:
            row = Version(id=1, schema_version=cls.CURRENT_VERSION)
            session.add(row)
            session.commit()
        return row.schema_version

    @classmethod
    def _set_version(cls, session: Session, version: int) -> int:
        session.execute(
            update(Version).where(Version.id == 1).values(schema_version=version)
        )
        session.commit()
        return version

    @classmethod
    def _run_migrations(cls, session: Session, prev_version: int, version: int):
        assert prev_version <= version
        raise RuntimeError(
            f"TODO: migration from {prev_version} to {version} not supported yet"
        )

    def _open_db_with_migrations(self, allow_migration: bool = False):
        assert self._db_file.exists()
        engine = self._db_create_engine(create=False)
        Base.metadata.create_all(engine, tables=[Version.__table__])
        with Session(engine) as session:
            actual_version = self._get_actual_version(session)

            if actual_version > self.CURRENT_VERSION:
                raise RuntimeError(
                    f"database schema {actual_version} is newer than current {self.CURRENT_VERSION}"
                )
            elif actual_version < self.MIN_VERSION:
                raise RuntimeError(
                    f"database schema {actual_version} is too old (< {self.MIN_VERSION})"
                )
            elif not allow_migration and actual_version != self.CURRENT_VERSION:
                raise RuntimeError(
                    f"database needs migration from {actual_version} to {self.CURRENT_VERSION}"
                )
            elif actual_version != self.CURRENT_VERSION:
                self._run_migrations(session, actual_version, self.CURRENT_VERSION)
                self._set_version(session, self.CURRENT_VERSION)
            return engine

    def create_unique_tag(self, tag: Tag):
        with Session(self._engine) as s:
            stmt = select(Tag).where(Tag.name == tag.name)
            obj = s.execute(stmt).scalar_one_or_none()
            if obj is not None:
                raise ValueError(f"Tag name {tag.name} already exists")
            s.add(Tag.from_plain(s, tag))
            s.commit()

    def get_or_create_tag(self, tag: DTag):
        with Session(self._engine) as s:
            Tag.from_plain(s, tag)

    def _session_get_tag(self, session: Session, tag: str) -> Tag:
        stmt = select(Tag).where(Tag.name == tag)
        tag_obj = session.execute(stmt).scalar_one_or_none()
        if tag_obj is None:
            raise ValueError(f"Tag name {tag} does not exist")
        return tag_obj

    def _session_get_tags(self, session: Session, tags: list[str]) -> list[Tag]:
        return [self._session_get_tag(session, tag) for tag in tags]

    def _session_tag_evaluation(
        self, session: Session, evaluation: Evaluation, tags: list[Tag]
    ):
        for tag in tags:
            EvaluationTag.get_or_create_unique(
                session,
                evaluation=evaluation,
                tag=tag,
            )

    def tag_evaluation(self, evaluation_id: int, tags: list[str]):
        with Session(self._engine) as s:
            evaluation = s.execute(
                select(Evaluation).where(Evaluation.id == evaluation_id)
            ).scalar_one_or_none()
            if evaluation is None:
                raise ValueError(f"evaluation id {evaluation_id} does not exist")
            tags_objs = self._session_get_tags(s, tags)
            self._session_tag_evaluation(s, evaluation, tags_objs)
            s.commit()

    def _session_record_evaluations(
        self, session: Session, evaluations: list[DEvaluation], tags: list[str]
    ):
        tags_objs = self._session_get_tags(session, tags)
        for evaluation in evaluations:
            eval_obj = Evaluation.from_plain(session, evaluation)
            session.add(eval_obj)
            self._session_tag_evaluation(session, eval_obj, tags_objs)

    def record_evaluation(self, evaluation: DEvaluation, tags: list[str]):
        with Session(self._engine) as s:
            self._session_record_evaluations(s, [evaluation], tags)
            s.commit()

    def record_evaluations(self, evaluations: list[DEvaluation], tags: list[str]):
        with Session(self._engine) as s:
            self._session_record_evaluations(s, evaluations, tags)
            s.commit()

    def get_payload_evaluations(self, payload: DPayload) -> dict[int, DEvaluation]:
        with Session(self._engine) as s:
            p = Payload.from_plain(s, payload)
            stmt = select(Evaluation).where(Evaluation.payload == p)
            rows = s.execute(stmt).scalars().all()
            return {e.id: e.as_plain(s) for e in rows}

    def get_operation_evaluations(
        self, operation: DOperation
    ) -> dict[int, DEvaluation]:
        with Session(self._engine) as s:
            obj = Operation.from_plain(s, operation)
            stmt = select(Evaluation).where(
                Evaluation.payload.has(Payload.operation == obj)
            )
            rows = s.execute(stmt).scalars().all()
            return {e.id: e.as_plain(s) for e in rows}

    def _session_get_filtered_tags_evaluations(
        self,
        session: Session,
        tags: list[str],
        full: bool = False,
        raw: bool = False,
        **kwargs: Any,
    ) -> dict[int, DEvaluation]:
        tags_ids = [self._session_get_tag(session, tag).id for tag in tags]
        assert len(tags_ids) >= 1
        obj_map = {k: self.from_plain(session, k, v) for k, v in kwargs.items()}
        conditions = [getattr(Payload, k) == v for k, v in obj_map.items()]
        sort_tag_id = tags_ids[0]
        et_filter = aliased(EvaluationTag)
        et_sort = aliased(EvaluationTag)
        stmt = (
            select(Evaluation)
            .options(
                selectinload(Evaluation.results).selectinload(Result.values),
            )
            .join(et_filter, et_filter.evaluation_id == Evaluation.id)
            .where(
                et_filter.tag_id.in_(tags_ids),
                *[Evaluation.payload.has(condition) for condition in conditions],
            )
            .join(
                et_sort,
                and_(
                    et_sort.evaluation_id == Evaluation.id,
                    et_sort.tag_id == sort_tag_id,
                ),
            )
            .group_by(Evaluation.id)
            .having(
                func.count(distinct(et_filter.tag_id)) == len(tags_ids),
            )
            .order_by(et_sort.created_at)
        )
        rows = session.execute(stmt).scalars().all()
        if raw:
            evals = {e.id: e for e in rows}
        else:
            evals = {e.id: e.as_plain(session, full=full) for e in rows}
        return evals

    def _session_get_filtered_tags_metric_evaluations(
        self,
        session: Session,
        tags: list[str],
        metric: str,
        full: bool = False,
        raw: bool = False,
        **kwargs: Any,
    ) -> dict[int, DEvaluation]:
        tags_ids = [self._session_get_tag(session, tag).id for tag in tags]
        assert len(tags_ids) >= 1
        obj_map = {k: self.from_plain(session, k, v) for k, v in kwargs.items()}
        conditions = [getattr(Payload, k) == v for k, v in obj_map.items()]
        sort_tag_id = tags_ids[0]
        et_filter = aliased(EvaluationTag)
        et_sort = aliased(EvaluationTag)
        stmt = (
            select(Evaluation)
            .options(
                selectinload(Evaluation.results).selectinload(Result.values),
            )
            .join(et_filter, et_filter.evaluation_id == Evaluation.id)
            .where(
                et_filter.tag_id.in_(tags_ids),
                *[Evaluation.payload.has(condition) for condition in conditions],
            )
            .join(Evaluation.results)
            .join(Result.values)
            .where(
                Result.metric == metric,
            )
            .join(
                et_sort,
                and_(
                    et_sort.evaluation_id == Evaluation.id,
                    et_sort.tag_id == sort_tag_id,
                ),
            )
            .group_by(Evaluation.id)
            .having(
                func.count(distinct(et_filter.tag_id)) == len(tags_ids),
            )
            .having(and_(func.count(Result.id) == 1, func.count(ResultValue.id) >= 1))
            .order_by(et_sort.created_at)
        )
        rows = session.execute(stmt).scalars().all()
        if raw:
            evals = {e.id: e for e in rows}
        else:
            evals = {e.id: e.as_plain(session, full=full) for e in rows}
        return evals

    def _session_get_filtered_evaluations(
        self, session: Session, full: bool = False, raw: bool = False, **kwargs: Any
    ) -> dict[int, DEvaluation]:
        obj_map = {k: self.from_plain(session, k, v) for k, v in kwargs.items()}
        conditions = [getattr(Payload, k) == v for k, v in obj_map.items()]
        stmt = (
            select(Evaluation)
            .options(
                selectinload(Evaluation.results).selectinload(Result.values),
            )
            .where(
                *[Evaluation.payload.has(condition) for condition in conditions],
            )
        )
        rows = session.execute(stmt).scalars().all()
        if raw:
            return {e.id: e for e in rows}
        else:
            return {e.id: e.as_plain(session, full=full) for e in rows}

    def get_filtered_evaluations(
        self, full: bool = False, raw: bool = False, **kwargs: Any
    ) -> dict[int, DEvaluation]:
        tags = []
        if "tags" in kwargs:
            tags = list(dict.fromkeys(kwargs["tags"]))
            del kwargs["tags"]
        with Session(self._engine) as s:
            if tags:
                return self._session_get_filtered_tags_evaluations(
                    s, tags, full, raw, **kwargs
                )
            else:
                return self._session_get_filtered_evaluations(s, full, raw, **kwargs)

    def delete_filtered_evaluations(
        self,
        **kwargs: Any,
    ) -> list[int]:
        tags = []
        if "tags" in kwargs:
            tags = list(dict.fromkeys(kwargs["tags"]))
            del kwargs["tags"]
        with Session(self._engine) as s:
            if tags:
                eval_objs = self._session_get_filtered_tags_evaluations(
                    s, tags, raw=True, **kwargs
                )
            else:
                eval_objs = self._session_get_filtered_evaluations(
                    s, raw=True, **kwargs
                )
            for eval_obj in eval_objs.values():
                s.delete(eval_obj)
            s.commit()
            return list(eval_objs)

    def get_filtered_metric_evaluations(
        self,
        metric: str = "elapsed",
        full: bool = False,
        raw: bool = False,
        **kwargs: Any,
    ) -> dict[int, DEvaluation]:
        tags = []
        if "tags" in kwargs:
            tags = list(dict.fromkeys(kwargs["tags"]))
            del kwargs["tags"]
        with Session(self._engine) as s:
            return self._session_get_filtered_tags_metric_evaluations(
                s, tags, metric, full, raw, **kwargs
            )

    def get_filtered_operations(self, **kwargs: Any) -> dict[int, DOperation]:
        with Session(self._engine) as s:
            conditions = [getattr(Operation, k) == v for k, v in kwargs.items()]
            stmt = select(Operation).where(
                *conditions,
            )
            rows = s.execute(stmt).scalars().all()
            return {o.id: o.as_plain(s) for o in rows}

    def _session_get_filtered_tags(
        self, session: Session, raw: bool = False, **kwargs: Any
    ) -> dict[int, DTag]:
        conditions = [getattr(Tag, k) == v for k, v in kwargs.items()]
        stmt = select(Tag).where(
            *conditions,
        )
        rows = session.execute(stmt).scalars().all()
        if raw:
            return {t.id: t for t in rows}
        else:
            return {t.id: t.as_plain(session) for t in rows}

    def get_filtered_tags(self, raw: bool = False, **kwargs: Any) -> dict[int, DTag]:
        with Session(self._engine) as s:
            return self._session_get_filtered_tags(s, raw=raw, **kwargs)

    def delete_filtered_tags(self, **kwargs: Any) -> list[int]:
        with Session(self._engine) as s:
            tag_objs = self._session_get_filtered_tags(s, raw=True, **kwargs)
            for obj in tag_objs.values():
                s.delete(obj)
            s.commit()
            return list(tag_objs)

    @classmethod
    def from_plain(cls, session: Session, target_cls: str, value: Any) -> Any:
        objcls = cls._class_map[target_cls.lower()]
        return objcls.from_plain(session, value)

    def get_from_digest(self, target_cls: str, digest: str) -> Any:
        with Session(self._engine) as s:
            objcls = self._class_map[target_cls.lower()]
            stmt = select(objcls).where(objcls.digest == digest)
            rows = s.execute(stmt).scalars().all()
            assert len(rows) == 1
            return rows[0].as_plain(s)

    def dump_dict(
        self, key: str, payload: dict, file: TextIO = sys.stdout, format: str = "yaml"
    ):
        dump_plain(payload=payload, key=key, file=file, format=format)

    def dump_all(
        self, file: TextIO = sys.stdout, format: str = "yaml", verbose: bool = False
    ):
        return self.dump_filtered(file, format, verbose)

    def dump_filtered(
        self,
        file: TextIO = sys.stdout,
        format: str = "yaml",
        verbose: bool = False,
        **kwargs: Any,
    ):
        payload = self.to_dict(verbose=verbose, **kwargs)
        self.dump_dict("evaluations", payload, file, format)

    def dump_tags(
        self,
        file: TextIO = sys.stdout,
        format: str = "yaml",
        verbose: bool = False,
        **kwargs: Any,
    ):
        with Session(self._engine) as session:
            stmt = select(Tag)
            objs = session.execute(stmt).scalars().all()
            payload = {
                "tags": [
                    dict(
                        id=tag.id,
                        created_at=deserialize_iso(tag.created_at),
                        updated_at=deserialize_iso(tag.updated_at),
                        **tag.as_plain(session).to_dict(),
                    )
                    if verbose
                    else tag.name
                    for tag in objs
                ]
            }
        self.dump_dict("tags", payload, file, format)

    def to_dict(self, verbose: bool = False, **kwargs: Any) -> dict:
        with Session(self._engine) as session:
            tag_name = None
            if "tag" in kwargs:
                tag_name = kwargs["tag"]
                del kwargs["tag"]
            obj_map = {k: self.from_plain(session, k, v) for k, v in kwargs.items()}
            conditions = [getattr(Payload, k) == v for k, v in obj_map.items()]
            stmt = (
                select(Evaluation)
                .options(
                    selectinload(Evaluation.results).selectinload(Result.values),
                )
                .join(Evaluation.tag_links)
                .join(EvaluationTag.tag)
                .where(
                    *([Tag.name == tag_name] if tag_name is not None else []),
                    *[Evaluation.payload.has(condition) for condition in conditions],
                )
            )
            evaluations = session.execute(stmt).scalars().all()
            payload = {
                "evaluations": [
                    {
                        "tags": [
                            dict(
                                name=l.tag.name,
                                created_at=deserialize_iso(l.tag.created_at),
                                updated_at=deserialize_iso(l.tag.updated_at),
                            )
                            if verbose
                            else l.tag.name
                            for l in e.tag_links
                        ],
                        **(
                            dict(
                                id=e.id,
                                created_at=deserialize_iso(e.created_at),
                                updated_at=deserialize_iso(e.updated_at),
                            )
                            if verbose
                            else {}
                        ),
                        "payload": {
                            **(
                                dict(
                                    id=e.payload.id,
                                    created_at=deserialize_iso(e.payload.created_at),
                                    updated_at=deserialize_iso(e.payload.updated_at),
                                )
                                if verbose
                                else {}
                            ),
                            **dict(
                                platform=[
                                    e.payload.platform.hostname,
                                    e.payload.platform.system,
                                    e.payload.platform.target,
                                ],
                                compiler=[
                                    e.payload.compiler.name,
                                    e.payload.compiler.version,
                                    e.payload.compiler.target,
                                    e.payload.compiler.threads,
                                    e.payload.compiler.backend,
                                ],
                                operation=[
                                    e.payload.operation.name,
                                    e.payload.operation.clsname,
                                    e.payload.operation.clsargs,
                                ],
                                schedule=[
                                    e.payload.schedule.clsname,
                                    e.payload.schedule.clsargs,
                                ],
                            ),
                        },
                        **dict(
                            code=e.code,
                            msg=e.msg,
                        ),
                        "results": [
                            {
                                **(
                                    dict(
                                        id=r.id,
                                        created_at=deserialize_iso(r.created_at),
                                        updated_at=deserialize_iso(r.updated_at),
                                    )
                                    if verbose
                                    else {}
                                ),
                                **dict(
                                    metric=r.metric,
                                ),
                                "values": [
                                    (
                                        dict(id=v.id, value=v.value)
                                        if verbose
                                        else v.value
                                    )
                                    for v in r.values
                                ],
                            }
                            for r in e.results
                        ],
                    }
                    for e in evaluations
                ]
            }
            return payload
