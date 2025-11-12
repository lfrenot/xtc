from concurrent.futures import Future, ThreadPoolExecutor
import itertools
import multiprocessing

from xtc.itf.schd import Schedule
from xtc.itf.back import Backend
from xtc.itf.graph import Graph

from xtc.dbs.evaluations import (
    DBEvaluator,
    EvaluationsDB,
)

def _get_matmul_graph(i: int, j: int, k: int, dtype: str = "float32") -> Graph:
    import xtc.graphs.xtc.op as O
    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="O")
    return gb.graph

def _get_matmul_schedule(impl: Backend, ir: int, jr: int, ku: int) -> Schedule:
    scheduler = impl.get_scheduler()
    scheduler.interchange(["k", "i", "j"])
    scheduler.vectorize(["j"])
    scheduler.unroll({"i": ir, "k": ku})
    return scheduler.schedule()

def test_evaluate_db(tmpdir: str):
    db = EvaluationsDB(f"{tmpdir}/db_test_dump", force_create=True)
    from xtc.backends.tvm import Backend

    IR_sizes = [2, 4, 8]
    JR_sizes = [16, 32, 64]
    KU_sizes = [8]
    samples = list(itertools.product(IR_sizes, JR_sizes, KU_sizes))

    def generate_evaluator(idx: int, sample: tuple[int,...], cached: bool = False):
        ir, jr, ku = sample
        graph = _get_matmul_graph(ir, jr, 512)
        backend = Backend(graph)
        schedule = _get_matmul_schedule(backend, ir, jr, ku)
        compiler = backend.get_compiler(
            shared_lib=True,
            dump_file=f"matmul_{ir}_{jr}_{ku}",
        )
        module = compiler.compile(schedule)
        evaluator = DBEvaluator(
            db,
            graph=graph,
            compiler=compiler,
            schedule=schedule,
            module=module,
            cached=cached,
        )
        return idx, sample, evaluator

    results = {}
    def future_callback(future: Future):
        idx, sample, evaluator = future.result()
        results[idx] = [sample, evaluator]

    jobs = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for idx, sample in enumerate(samples):
            future = executor.submit(
                generate_evaluator,
                idx=idx,
                sample=sample,
            )
            future.add_done_callback(future_callback)

    with ThreadPoolExecutor(max_workers=1) as executor:
        for _, evaluator in results.values():
            executor.submit(
                evaluator.evaluate,
            )

    evaluations = db.get_filtered_evaluations()
    assert len(evaluations) == len(samples)
    for evaluation in evaluations.values():
        assert len(evaluation.results) == 1
        digest = evaluation.payload.operation.get_digest()
        operation = db.get_from_digest("operation", digest)
        print(digest, operation, min(evaluation.results[0].values))

    # Test evaluation cache
    _, _, evaluator = generate_evaluator(-1, samples[0], cached=True)
    values, _, _ = evaluator.evaluate()
    evaluations = db.get_filtered_evaluations(operation=results[0][1].operation)
    cached_evaluation = evaluations.popitem()[1]
    assert cached_evaluation.results[0].values == values
