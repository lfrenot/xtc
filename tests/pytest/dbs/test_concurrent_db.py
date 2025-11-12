from typing import Any
from xtc.dbs.evaluations import (
    EvaluationsDB,
    get_compiler_xtc,
    get_platform,
    get_schedule_strategy,
    get_operation_graph,
    get_result,
    get_payload,
    get_evaluation,
)

def _get_matmul_graph():
    import xtc.graphs.xtc.op as O
    i, j, k, dtype = 128, 128, 128, "float32"
    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="O")
    return gb.graph


def test_concurrent_db(tmpdir: str):
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, Future
    import random
    db = EvaluationsDB(f"{tmpdir}/db_test_concurrent", force_create=True)
    compiler = get_compiler_xtc(backend="tvm")
    platform = get_platform()

    import xtc.search.strategies as S
    graph = _get_matmul_graph()
    operation = get_operation_graph(graph, {})
    strategy = S.Strategy_P1(graph=graph)
    samples = list(strategy.sample(num=128))

    def job_start(idx: int, sample: Any):
        print(f"Job start {idx}: {sample}")
    def job_end(idx: int, sample: Any):
        print(f"Job end {idx}: {sample}")
    def generate_evaluation(idx: int, sample: Any):
        schedule = get_schedule_strategy(strategy, {}, sample)
        payload = get_payload(
            platform=platform,
            compiler=compiler,
            operation=operation,
            schedule=schedule,
        )
        n1 = random.randint(1, 3)
        n2 = random.randint(1, 3)
        elapsed = [random.random() for _ in range(n1)]
        peak = [random.random() for _ in range(n1)]
        io = [random.random() for _ in range(n2)]
        evaluations = [
            get_evaluation(
                payload=payload,
                code=0,
                msg="",
                results=[
                    get_result("elapsed", elapsed),
                    get_result("peak", peak),
                ],
            ),
            get_evaluation(
                payload=payload,
                code=0,
                msg="",
                results=[
                    get_result("io", io),
                ],
            ),
        ]
        db.record_evaluations(evaluations)
        return idx, sample

    def future_callback(future: Future):
        idx, sample = future.result()
        job_end(idx, sample)

    jobs = multiprocessing.cpu_count()
    print(f"Start tasks...")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for idx, sample in enumerate(samples):
            job_start(idx, sample)
            future = executor.submit(
                generate_evaluation,
                idx=idx,
                sample=sample,
            )
            future.add_done_callback(future_callback)
    print(f"Completed tasks.")
    evaluations = db.get_operation_evaluations(operation)
    assert len(evaluations) == 2*len(samples)
    for evaluation in evaluations:
        print(evaluation)
