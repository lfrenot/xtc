from xtc.dbs.evaluations import (
    EvaluationsDB,
    get_compiler_xtc,
    get_platform,
    get_schedule_strategy,
    get_operation_graph,
    get_result,
    get_payload,
    get_evaluation,
    get_tag,
)

def _get_matmul_graph():
    import xtc.graphs.xtc.op as O
    i, j, k, dtype = 128, 128, 128, "float32"
    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="O")
    return gb.graph


def test_dump_db(tmpdir: str):
    db = EvaluationsDB(f"{tmpdir}/db_test_dump", force_create=True)
    compiler1 = get_compiler_xtc(backend="tvm")
    compiler2 = get_compiler_xtc(backend="mlir")
    platform = get_platform()

    import xtc.search.strategies as S
    tag = get_tag("tag", "session tag")
    tag1 = get_tag("tag1", "first tag")
    tag2 = get_tag("tag2", "second tag")
    db.create_unique_tag(tag)
    db.get_or_create_tag(tag1)
    db.get_or_create_tag(tag2)
    graph = _get_matmul_graph()
    operation = get_operation_graph(graph, {})
    strategy = S.Strategy_P1(graph=graph)
    sample = list(strategy.sample(num=1))[0]
    schedule = get_schedule_strategy(strategy, {}, sample)
    results = [
        get_result(metric="elapsed", values=[1, 1.0, 1.1]),
        get_result(metric="l1miss", values=[10, 11, 12]),
        get_result(metric="peak", values=[0.8]),
        get_result(metric="elapsed", values=[0.9]),
        get_result(metric="elapsed", values=[0.8, 0.9, 1.0]),
    ]
    payload1 = get_payload(
        platform=platform,
        compiler=compiler1,
        operation=operation,
        schedule=schedule,
    )
    payload2 = get_payload(
        platform=platform,
        compiler=compiler2,
        operation=operation,
        schedule=schedule,
    )
    evaluations = [
        get_evaluation(
            payload=payload1,
            code=0,
            msg="",
            results=results[:2],
        ),
        get_evaluation(
            payload=payload1,
            code=0,
            msg="",
            results=results[2:4],
        ),
        get_evaluation(
            payload=payload2,
            code=0,
            msg="",
            results=results[4:],
        ),
    ]
    db.record_evaluation(evaluations[0], [tag.name, tag1.name])

    # Test db re-open
    db = EvaluationsDB(f"{tmpdir}/db_test_dump")
    db.record_evaluations(evaluations[1:], [tag.name, tag2.name])

    db.dump_all(format="jsonl", verbose=True)
    db.dump_all(format="yaml")
    db.dump_all(format="json")
    db.dump_all(format="python")

    evaluations = db.get_payload_evaluations(payload1)
    assert len(evaluations) == 2
    for evaluation in evaluations:
        print(evaluation)
    evaluations = db.get_payload_evaluations(payload2)
    assert len(evaluations) == 1
    for evaluation in evaluations:
        print(evaluation)
    evaluations = db.get_operation_evaluations(operation)
    assert len(evaluations) == 3
    for evaluation in evaluations:
        print(evaluation)
    evaluations = db.get_filtered_evaluations(compiler=compiler1)
    assert len(evaluations) == 2
    for evaluation in evaluations:
        print(evaluation)

    evaluations = db.get_filtered_evaluations(compiler=compiler1, tags=[tag1.name])
    assert len(evaluations) == 1

    evaluations = db.get_filtered_evaluations(tags=[tag2.name])
    assert len(evaluations) == 2
