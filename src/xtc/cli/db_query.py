#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import logging
import argparse
from argparse import Namespace as NS
from typing import Any
import re

from xtc.runtimes.host import runtime
from xtc.utils.math import mulall
from xtc.cli.display_results import display_results
from xtc.utils.dump import dump_plain
from xtc.dbs.evaluations import (
    EvaluationsDB,
    Evaluation,
    get_compiler_xtc,
    get_operation_from_artifacts,
    get_platform,
    get_tag,
)

logger = logging.getLogger(__name__)


def dump_objs(objs: dict, format: str = "plain", kwargs_list: list[dict] = []):
    if format == "plain":
        if kwargs_list:
            for (id, obj), kwargs in zip(objs.items(), kwargs_list):
                print(f"{id}: {obj}: {kwargs}")
        else:
            for id, obj in objs.items():
                print(f"{id}: {obj}")
    else:
        if kwargs_list:
            dump_objs = [
                {**dict(_id=id), **obj.to_dict(), **kwargs}
                for (id, obj), kwargs in zip(objs.items(), kwargs_list)
            ]
        else:
            dump_objs = [{**dict(_id=id), **obj.to_dict()} for id, obj in objs.items()]
        dump_plain(
            dump_objs,
            format=format,
        )


def dump_operations(args: NS):
    db = EvaluationsDB(args.db)
    objs = db.get_filtered_operations(
        **(dict(name=args.operator) if args.operator else {}),
    )
    dump_objs(objs, args.format)


def dump_tags(args: NS):
    db = EvaluationsDB(args.db)
    if args.tags:
        objs = {}
        for tag in args.tags:
            objs.update(db.get_filtered_tags(name=tag))
    else:
        objs = db.get_filtered_tags()
    if args.tags_strs:
        regex = re.compile(r"|".join([re.escape(t) for t in args.tags_strs]))
        objs = {k: obj for k, obj in objs.items() if regex.search(obj.name)}
    dump_objs(objs, args.format)


def delete_tag(args: NS):
    db = EvaluationsDB(args.db)
    assert args.tag
    answer = input(f"Delete Tag: {args.tag}\nContinue [N/y]?")
    if answer.lower() not in ["y", "yes"]:
        print("Aborted.")
    ids = db.delete_filtered_tags(name=args.tag)
    print(f"Deleted {len(ids)} Tags.")


def create_tags(args: NS):
    db = EvaluationsDB(args.db)
    for tag_name in args.new_tags:
        plain_tag = get_tag(tag_name)
        logger.info("creating tag: %s", plain_tag)
        db.create_unique_tag(plain_tag)


def get_operation_flop(args: NS) -> int | None:
    if args.operator and args.op_name and args.dtype:
        operation = get_operation_from_artifacts(
            args.operator, args.op_name, args.dtype
        )
        return mulall(list(operation.clsargs["dims"].values()))
    return None


def get_evaluations_filters(args: NS, tags: list[str] = []) -> dict[str, Any]:
    platform = (
        get_platform()
        if args.machine is None
        else get_platform(args.machine, args.system, args.target)
    )
    compiler = get_compiler_xtc(args.backend, args.target, args.threads)
    operation = None
    if args.operator and args.op_name and args.dtype:
        operation = get_operation_from_artifacts(
            args.operator, args.op_name, args.dtype
        )

    all_tags = (args.tags if args.tags else []) + tags
    return dict(
        **(dict(tags=all_tags) if all_tags else {}),
        platform=platform,
        compiler=compiler,
        **(dict(operation=operation) if operation is not None else {}),
    )


def get_evaluations(
    db: EvaluationsDB, args: NS, full: bool = False
) -> dict[int, Evaluation]:
    if args.topk:
        objs = db.get_filtered_metric_evaluations(
            metric="elapsed",
            **get_evaluations_filters(args),
            full=full,
            raw=args.raw,
        )
        if args.raw:
            sort_f = lambda e: min([v.value for v in e.results[0].values])
        else:
            sort_f = lambda e: min(e.results[0].values)
        objs = dict(sorted(objs.items(), key=lambda x: sort_f(x[1]))[: args.topk])
    else:
        objs = db.get_filtered_evaluations(
            **get_evaluations_filters(args),
            full=full,
            raw=args.raw,
        )
    return objs


def list_evaluations(args: NS):
    db = EvaluationsDB(args.db)
    full = args.verbose
    flop = get_operation_flop(args)
    if args.peak and flop is None:
        full = True
    objs = get_evaluations(db, args, full=full)
    kwargs_list = []
    if args.peak:
        ops_time = [min(e.results[0].values) for e in objs.values()]
        if flop is None:
            ops_flop = [
                mulall(list(e.payload.operation.clsargs["dims"].values()))
                for e in objs.values()
            ]
            kwargs_list = [
                {"peak": flop / time / args.flops}
                for flop, time in zip(ops_flop, ops_time)
            ]
        else:
            kwargs_list = [{"peak": flop / time / args.flops} for time in ops_time]
    dump_objs(objs, args.format, kwargs_list)


def dump_evaluations(args: NS):
    db = EvaluationsDB(args.db)
    objs = get_evaluations(db, args, full=True)
    dump_objs(objs, "plain")


def load_evaluations(args: NS):
    from xtc.dbs.evaluations import (
        Platform,
        Operation,
        Compiler,
        Schedule,
        Payload,
        Result,
        Evaluation,
    )

    assert args.tags, f"must give some tags when loading evaluations"
    db = EvaluationsDB(args.db)
    evaluations = []
    with open(args.dump_file) as inf:
        for id_plain in inf.readlines():
            obj_str = id_plain.split(": ", 1)[1]
            evaluation = eval(
                obj_str,
                {},
                dict(
                    Platform=Platform,
                    Compiler=Compiler,
                    Operation=Operation,
                    Schedule=Schedule,
                    Payload=Payload,
                    Result=Result,
                    Evaluation=Evaluation,
                ),
            )
            evaluations.append(evaluation)
    logger.info("loading %d evaluations with tags: %s", len(evaluations), args.tags)
    db.record_evaluations(evaluations, args.tags)


def delete_evaluations(args: NS):
    db = EvaluationsDB(args.db)
    filter = get_evaluations_filters(args)
    answer = input(f"Delete Evaluations matching: {filter}\nContinue [N/y]?")
    if answer.lower() not in ["y", "yes"]:
        print("Aborted.")
    ids = db.delete_filtered_evaluations(**get_evaluations_filters(args))
    print(f"Deleted {len(ids)} Evaluations.")


def tag_evaluations(args: NS):
    db = EvaluationsDB(args.db)
    objs = get_evaluations(db, args)
    for idx, obj in objs.items():
        logger.info("taging evaluation: %d: %s", idx, obj)
        db.tag_evaluation(idx, args.new_tags)


def display_evaluations(args: NS):
    db = EvaluationsDB(args.db)
    results = []
    flop = get_operation_flop(args)
    for idx, result_tag in enumerate(args.result_tags):
        raw = True
        if not flop:
            raw = False
        objs = db.get_filtered_metric_evaluations(
            metric="elapsed",
            full=True,
            raw=raw,
            **get_evaluations_filters(args, tags=[result_tag]),
        )
        logger.debug("Results for %s: num: %d", result_tag, len(objs))
        if raw:
            values = [
                min([v.value for v in eval.results[0].values]) for eval in objs.values()
            ]
        else:
            values = [min(eval.results[0].values) for eval in objs.values()]
        if not flop:
            ops_flop = [
                mulall(list(eval.payload.operation.clsargs["dims"].values()))
                for eval in objs.values()
            ]
            values = [
                (op_flop / time) / args.flops for op_flop, time in zip(ops_flop, values)
            ]
        else:
            values = [(flop / time) / args.flops for time in values]
        result = NS(Y=values, label=f"{result_tag}")
        results.append(result)
    display_results(results, args)


def main():
    default_dtype = "float32"
    default_target = "native"
    default_system = "native"
    default_backend = "tvm"
    default_threads = 1
    default_db = "results_db"
    default_format = "jsonl"

    parser = argparse.ArgumentParser(
        description="Query results database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="object", required=True)

    common = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_argument("--db", type=str, default=default_db, help="database dir name")
    common.add_argument(
        "--verbose", action="store_true", help="dump database in verbose mode"
    )
    common.add_argument("--raw", action="store_true", help="dump database in raw mode")
    common.add_argument(
        "--format", type=str, default=default_format, help="dump database format"
    )
    common.add_argument(
        "--quiet", action="store_true", help="quiet mode, only results on output"
    )
    common.add_argument("--debug", action="store_true", help="debug mode")
    common.add_argument("--debug-sql", action="store_true", help="dump sql queries")

    mach_filter = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mach_filter.add_argument("--machine", type=str, help="machine or current host")
    mach_filter.add_argument(
        "--system", type=str, default=default_system, help="machine system"
    )
    mach_filter.add_argument(
        "--target", type=str, default=default_target, help="machine target"
    )

    comp_filter = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    comp_filter.add_argument(
        "--backend", type=str, default=default_backend, help="compiler backend"
    )
    comp_filter.add_argument(
        "--threads", type=int, default=default_threads, help="target threads"
    )

    eval_filter = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_filter.add_argument("--operator", type=str, help="Operator type")
    eval_filter.add_argument("--tag", type=str, help="dump result for given tag")
    eval_filter.add_argument(
        "--tags", type=str, nargs="+", help="dump result for given tags"
    )

    subparsers.add_parser(
        "operations",
        parents=[common, mach_filter, comp_filter, eval_filter],
        help="query operations",
    )

    t_parser = subparsers.add_parser("tags", help="manage tags")
    t_subparsers = t_parser.add_subparsers(dest="action")
    t_l_parser = t_subparsers.add_parser(
        "list",
        parents=[common, mach_filter, comp_filter, eval_filter],
        help="list tags",
    )
    t_l_parser.add_argument(
        "tags_strs", nargs="*", type=str, help="tags substrings to list or all"
    )

    t_c_parser = t_subparsers.add_parser("create", parents=[common], help="create tags")
    t_c_parser.add_argument(
        "new_tags", nargs="*", type=str, help="tags names to create"
    )

    t_d_parser = t_subparsers.add_parser("delete", parents=[common], help="delete tags")
    t_d_parser.add_argument("tag", type=str, help="tag to delete")

    op_filter = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    op_filter.add_argument("--op-name", type=str, help="operation name")
    op_filter.add_argument(
        "--dtype",
        type=str,
        default=default_dtype,
        choices=["float32", "float64"],
        help="data type",
    )
    op_filter.add_argument("--topk", type=int, help="show topk best")

    flop_args = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    flop_args.add_argument(
        "--flops", type=float, help="flop/sec for the machine/dtype or estimated"
    )

    e_parser = subparsers.add_parser("evaluations", help="manage evaluations")
    e_subparsers = e_parser.add_subparsers(dest="action", required=True)
    e_l_parser = e_subparsers.add_parser(
        "list",
        parents=[common, mach_filter, comp_filter, eval_filter, op_filter, flop_args],
        help="list evaluations",
    )
    e_l_parser.add_argument(
        "--best", action="store_true", help="show best evaluations only"
    )
    e_l_parser.add_argument("--peak", action="store_true", help="show peak perf")

    e_t_parser = e_subparsers.add_parser(
        "tag",
        parents=[common, mach_filter, comp_filter, eval_filter, op_filter],
        help="tag evaluations",
    )
    e_t_parser.add_argument("new_tags", nargs="*", type=str, help="tags to apply")

    e_d_parser = e_subparsers.add_parser(
        "dump",
        parents=[common, mach_filter, comp_filter, eval_filter, op_filter],
        help="dump evaluations",
    )

    e_a_parser = e_subparsers.add_parser(
        "load", parents=[common, eval_filter], help="load evaluations"
    )
    e_a_parser.add_argument("dump_file", type=str, help="load file")

    e_r_parser = e_subparsers.add_parser(
        "delete",
        parents=[common, mach_filter, comp_filter, eval_filter, op_filter],
        help="delete evaluations",
    )

    d_parser = subparsers.add_parser(
        "display",
        parents=[common, mach_filter, comp_filter, eval_filter, op_filter, flop_args],
        help="display evaluations",
    )
    d_parser.add_argument("--title", type=str, help="Figure title")
    d_parser.add_argument("--output", type=str, help="Save figure to file")
    d_parser.add_argument(
        "--pmf", action=argparse.BooleanOptionalAction, default=True, help="draw PMF"
    )
    d_parser.add_argument(
        "--cdf", action=argparse.BooleanOptionalAction, default=False, help="draw CDF"
    )
    d_parser.add_argument(
        "--rcdf", action=argparse.BooleanOptionalAction, default=True, help="draw RCDF"
    )
    d_parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show figure",
    )
    d_parser.add_argument("result_tags", nargs="+", help="selection tags for results")

    args = parser.parse_args()

    logging.basicConfig()
    if not args.quiet:
        logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)
    if args.debug_sql:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    if args.object == "tags" and args.action == "delete":
        delete_tag(args)
        raise SystemExit()

    if args.object == "tags" and args.action == "create":
        create_tags(args)
        raise SystemExit()

    if hasattr(args, "flops") and hasattr(args, "dtype") and args.flops is None:
        args.flops = runtime.evaluate_flops(args.dtype)
        logger.debug("Host Machine evaluated flops: %.3f", args.flops)

    if args.tag:
        if args.tags:
            raise ValueError(f"options --tag and --tags are incompatible")
        args.tags = [args.tag]

    if args.object == "tags":
        assert args.action == "list"
        dump_tags(args)
        raise SystemExit()
    elif args.object == "operations":
        dump_operations(args)
        raise SystemExit()
    elif args.object == "evaluations":
        if args.action == "list":
            if args.best and not args.topk:
                args.topk = 1
            list_evaluations(args)
        elif args.action == "dump":
            dump_evaluations(args)
        elif args.action == "load":
            load_evaluations(args)
        elif args.action == "delete":
            delete_evaluations(args)
        else:
            assert args.action == "tag"
            tag_evaluations(args)
    elif args.object == "display":
        display_evaluations(args)


if __name__ == "__main__":
    main()
