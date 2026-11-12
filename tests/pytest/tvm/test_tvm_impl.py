import pytest
from pathlib import Path
from tvm_utils import requires_tvm, matmul_impl

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)

@requires_tvm
def test_split_allows_tiling_outer_axis():
    from xtc.schedules.descript import descript_scheduler

    # Odd extents exercise the partial tile tails; evaluate() validates outputs
    # against the reference implementation.
    impl = matmul_impl(51, 65, 17, DTYPE, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {"j#2": {}, "i#3": {}},
            "i[12:]": {"j#2": {}, "i#2": {}},
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert schedule_str.count("split(j, factors=[None, 2])") == 1
    assert schedule_str.count("sch.reorder(i, j0, i0)") == 2
    assert "loop_partition_rebased" in schedule_str
    assert isinstance(impl.evaluate(schedule), float)


@requires_tvm
def test_split_hoists_multiple_non_split_axis_tiles():
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend
    from xtc.backends.tvm.TVMScheduler import tvm_plan_split_tiles
    from xtc.schedules.descript import descript_scheduler

    inp = O.tensor((17, 19, 23), "float32")
    with O.graph(name="relu_graph") as gb:
        O.relu(inp, name="C")
    impl = TVMBackend(gb.graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {
                "j#4": {},
                "j#2": {},
                "k#6": {},
                "k#3": {},
            },
            "i[12:]": {
                "j#4": {},
                "j#2": {},
                "k#6": {},
                "k#3": {},
            },
        },
    )

    transformed = tvm_plan_split_tiles(sch.get_loop_nest())
    root = transformed.root_node
    assert root is not None
    assert root.tiles == {
        "j": {"j0": 4, "j1": 2},
        "k": {"k0": 6, "k1": 3},
    }
    assert len(root.children) == 2
    for branch in root.children:
        assert "j" not in branch.tiles
        assert "k" not in branch.tiles
        assert branch.interchange == ["i", "j0", "j1", "k0", "k1"]

    schedule_str = str(sch.schedule())
    assert schedule_str.count(" = sch.split(j, factors=[") == 1
    assert schedule_str.count(" = sch.split(k, factors=[") == 1
    assert isinstance(impl.evaluate(sch.schedule()), float)


@requires_tvm
def test_conv2d_tiles_reduction_and_spatial_axes_across_partition_tails():
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend
    from xtc.schedules.descript import descript_scheduler

    # Odd spatial/reduction extents make every tiled axis exercise a tail. The
    # two h partitions also force reduction initialization to be valid per branch.
    inp = O.tensor((1, 9, 6, 5), DTYPE, name="A")
    weight = O.tensor((3, 2, 5, 5), DTYPE, name="W")
    with O.graph(name="conv2d_tiled") as gb:
        O.conv2d(inp, weight, name="C")
    impl = TVMBackend(gb.graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["b", "h", "w", "f", "r", "s", "c"],
        spec={
            "b": {},
            "w": {},
            "f": {},
            "r": {},
            "s": {},
            "c": {},
            "h[:3]": {
                "w#2": {},
                "f#2": {},
                "r#2": {},
                "s#2": {},
                "c#4": {},
                "c#2": {},
            },
            "h[3:]": {
                "w#2": {},
                "f#2": {},
                "r#2": {},
                "s#2": {},
                "c#4": {},
                "c#2": {},
            },
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert "loop_partition_rebased" in schedule_str
    assert schedule_str.count("sch.split(r, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(s, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(c, factors=[None, 2, 2])") == 1
    assert schedule_str.count("sch.split(w, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(f, factors=[None, 2])") == 1
    assert isinstance(impl.evaluate(schedule), float)

    # XTC does not expose scheduled TIR through its public backend API, so
    # inspect the resulting TVM TIR Schedule directly rather than just the script.
    from xtc.backends.tvm.TVMOpsCompiler import TVMExprCompiler

    tir = TVMExprCompiler(impl._tvm_base).generate().schedule(schedule).dumps()
    first_reduction_loop = tir.index("for r_0")
    assert tir.index('sblock("C_h0_init")') < first_reduction_loop
    assert tir.index('sblock("C_h1_init")') < first_reduction_loop


@requires_tvm
def test_split_allows_tiled_reduction_axis_with_tails():
    from xtc.schedules.descript import descript_scheduler

    # Both the partition boundary (12 of 19 rows) and reduction extent (23
    # elements tiled by 6) have tails. The reduction tile must be initialized
    # correctly in each partition for the result to match the reference.
    impl = matmul_impl(19, 11, 23, DTYPE, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {"k#6": {}},
            "i[12:]": {"k#6": {}},
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert schedule_str.count("loop_partition_rebased") == 1
    assert schedule_str.count("sch.split(k, factors=[None, 6])") == 1
    # Initializers are decomposed after partitioning so each partition's
    # reduction receives a correctly placed initialization.
    assert schedule_str.index("loop_partition_rebased") < schedule_str.index(
        "decompose_reduction_initializers"
    )
    assert isinstance(impl.evaluate(schedule), float)


@requires_tvm
def test_full_conv_split_schedule_preserves_branch_tiles_and_annotations():
    import xtc.graphs.xtc.op as O
    from xtc.artifacts import get_operation
    from xtc.backends.tvm import TVMBackend
    from xtc.backends.tvm.TVMOpsCompiler import TVMExprCompiler
    from xtc.schedules.descript import descript_scheduler

    op = get_operation("conv2d", "ResNet18_01")
    n, h, w, f, r, s, c = (
        op["dims"][key] for key in ("n", "h", "w", "f", "r", "s", "c")
    )
    sh, sw = (op["params"][key] for key in ("SH", "SW"))
    inp = O.tensor((n, h + r - 1, w + s - 1, c), DTYPE, name="A")
    weight = O.tensor((r, s, c, f), DTYPE, name="W")
    with O.graph(name="conv2d_full_split") as gb:
        O.conv2d(inp, weight, stride=(sh, sw), name="O")

    impl = TVMBackend(gb.graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="O",
        abstract_dims=["b", "h", "w", "f", "r", "s", "c"],
        spec={
            "b": {},
            "h": {},
            "f": {},
            "r": {},
            "s": {},
            "c": {},
            "w[:12]": {
                "h#56": {},
                "w#6": {},
                "f#32": {},
                "r#3": {},
                "s#3": {},
                "c#3": {"unroll": True},
                "w#3": {"unroll": True},
                "f#16": {"vectorize": True},
            },
            "w[12:]": {
                "h#56": {},
                "w#50": {},
                "f#32": {},
                "r#3": {},
                "s#3": {},
                "c#3": {"unroll": True},
                "w#5": {"unroll": True},
                "f#16": {"vectorize": True},
            },
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert schedule_str.count("loop_partition_rebased") == 1
    for expected in (
        "sch.split(h, factors=[None, 56])",
        "sch.split(f, factors=[None, 2, 16])",
        "sch.split(r, factors=[None, 3])",
        "sch.split(s, factors=[None, 3])",
        "sch.split(c, factors=[None, 3])",
        "sch.split(w, factors=[None, 2, 3])",
        "sch.split(w, factors=[None, 10, 5])",
    ):
        assert expected in schedule_str
    assert "unroll" in schedule_str
    assert "vectorize" in schedule_str

    # The branch-specific tile sizes and loop annotations must survive lowering.
    tir = TVMExprCompiler(impl._tvm_base).generate().schedule(schedule).dumps()
    assert 'sblock("O_w0_init")' in tir
    assert 'sblock("O_w1_init")' in tir
    assert "w0_0, h_1, w0_1" in tir
    assert "w1_zero_0, h_1, w1_zero_1" in tir
    assert "for w0_2 in T.unroll(3)" in tir
    assert "for w1_zero_2 in T.unroll(5)" in tir
    assert "T.vectorized" in tir


@requires_tvm
def test_conv_split_allows_matching_outer_tiles_with_different_branch_tiling():
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend
    from xtc.schedules.descript import descript_scheduler

    inp = O.tensor((1, 9, 20, 5), DTYPE, name="A")
    weight = O.tensor((3, 2, 5, 5), DTYPE, name="W")
    with O.graph(name="conv2d_branch_tiles") as gb:
        O.conv2d(inp, weight, name="C")
    impl = TVMBackend(gb.graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["b", "h", "w", "f", "r", "s", "c"],
        spec={
            "b": {},
            "h": {},
            "f": {},
            "r": {},
            "s": {},
            "c": {},
            "w[:10]": {
                "h#2": {},
                "w#6": {},
                "f#2": {},
                "r#2": {},
                "s#2": {},
                "c#2": {},
                "f#1": {},
            },
            "w[10:]": {
                "h#2": {},
                "w#5": {},
                "r#2": {},
                "s#2": {},
                "c#2": {},
                "f#2": {},
            },
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert schedule_str.count("loop_partition_rebased") == 1
    assert schedule_str.count("sch.split(h, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(f, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(f0, factors=[None, 1])") == 1
    assert schedule_str.count("sch.split(r, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(s, factors=[None, 2])") == 1
    assert schedule_str.count("sch.split(c, factors=[None, 2])") == 1
    assert "sch.split(w, factors=[None, 6])" in schedule_str
    assert "sch.split(w, factors=[None, 5])" in schedule_str
    assert "sch.reorder(w, h0, w0, f0, r0, s0, c0, f1)" in schedule_str
    assert "sch.reorder(w, h0, w0, r0, s0, c0, f0)" in schedule_str
    assert isinstance(impl.evaluate(schedule), float)


@requires_tvm
def test_conv_split_rejects_mismatched_outer_tiles_with_branch_tiling():
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend
    from xtc.schedules.descript import descript_scheduler
    from xtc.schedules.exceptions import ScheduleValidationError

    inp = O.tensor((1, 9, 6, 5), DTYPE, name="A")
    weight = O.tensor((3, 2, 5, 5), DTYPE, name="W")
    with O.graph(name="conv2d_mismatched_branch_tiles") as gb:
        O.conv2d(inp, weight, name="C")
    sch = TVMBackend(gb.graph).get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["b", "h", "w", "f", "r", "s", "c"],
        spec={
            "b": {},
            "h": {},
            "f": {},
            "r": {},
            "s": {},
            "c": {},
            "w[:2]": {"h#2": {}, "w#2": {}, "f#2": {}, "w#1": {}},
            "w[2:]": {"h#4": {}, "w#2": {}, "f#2": {}, "w#2": {}},
        },
    )

    with pytest.raises(
        ScheduleValidationError, match=r"requires matching tiles|cannot tile axis"
    ):
        sch.schedule()


@requires_tvm
def test_split_rejects_mismatched_outer_axis_tile_sizes():
    from xtc.schedules.descript import descript_scheduler
    from xtc.schedules.exceptions import ScheduleValidationError

    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {"j#2": {}, "i#3": {}},
            "i[12:]": {"j#4": {}, "i#2": {}},
        },
    )
    with pytest.raises(ScheduleValidationError, match=r"requires matching tiles"):
        sch.schedule()


@requires_tvm
def test_matmul_split_nested():
    from xtc.schedules.descript import descript_scheduler

    # Odd matrix extents cover partition tails at both levels. evaluate() checks
    # the scheduled result against the reference implementation numerically.
    impl = matmul_impl(19, 11, 7, DTYPE, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:10]": {
                "i[:5]": {"j#11": {}, "k#7": {}},
                "i[5:]": {"j#11": {}, "k#7": {}},
            },
            "i[10:]": {"j#11": {}, "k#7": {}},
        },
    )

    schedule = sch.schedule()
    schedule_str = str(schedule)
    assert schedule_str.count("loop_partition_rebased") >= 2
    assert "sch.split(j, factors=[None, 11])" in schedule_str
    assert "sch.split(k, factors=[None, 7])" in schedule_str
    assert isinstance(impl.evaluate(schedule), float)


@requires_tvm
def test_split_rejects_nested_partition_of_outer_loop():
    from xtc.schedules.descript import descript_scheduler
    from xtc.schedules.exceptions import ScheduleValidationError

    impl = matmul_impl(19, 11, 7, DTYPE, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:10]": {
                "j[:5]": {"k": {}},
                "j[5:]": {"k": {}},
            },
            "i[10:]": {"j": {}, "k": {}},
        },
    )

    with pytest.raises(ScheduleValidationError, match="loop is outside that scope"):
        sch.schedule()


@requires_tvm
def test_split_reorder_keeps_k_outside_i():
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend
    from xtc.schedules.descript import descript_scheduler

    inp = O.tensor((19, 11, 7), DTYPE)
    with O.graph(name="relu_graph") as gb:
        O.relu(inp, name="C")
    impl = TVMBackend(gb.graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "k": {},
            "i[:10]": {"j": {}, "k": {}},
            "i[10:]": {"j": {}, "k": {}},
        },
    )

    schedule_str = str(sch.schedule())
    assert "sch.reorder(k, i, j)" in schedule_str


@requires_tvm
def test_split_rejects_tile_missing_from_branch():
    from xtc.schedules.descript import descript_scheduler
    from xtc.schedules.exceptions import ScheduleValidationError

    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {"j#2": {}, "i#3": {}},
            "i[12:]": {"i#2": {}},
        },
    )
    with pytest.raises(ScheduleValidationError, match=r"requires matching tiles"):
        sch.schedule()


@requires_tvm
def test_split_allows_tiling_partition_axis():
    from xtc.schedules.descript import descript_scheduler

    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["i", "j", "k"],
        spec={
            "j": {},
            "k": {},
            "i[:12]": {"i#3": {}},
            "i[12:]": {"i#2": {}},
        },
    )
    schedule = str(sch.schedule())
    assert "split(i, factors=[None, 3])" in schedule
    assert "split(i, factors=[None, 2])" in schedule


def sched_nop(sch):
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, j, k)"
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, i1, i2, j, j1, j2, k, k1)",
        "split(i, factors=[None, 16, 4])",
        "split(j, factors=[None, 1, 64])",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 16})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1", "j", "k", "j1", "k1", "i2", "j2"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, i1, j, k, j1, k1, i2, j2)",
        "vectorize(j2)",
        "fuse(i, i1)",
        "parallel(i1)",
    ]

def sched_tile3wc(sch):
    sch.tile("i", {"i1": 64, "i2": 32, "i3": 4})
    sch.tile("j", {"j1": 256, "j2": 64, "j3": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "j", "i1", "j1", "k", "i2", "j2", "k1", "i3", "j3"])
    sch.parallelize(["i", "j"])
    sch.unroll({"k1": 13, "i3": 4, "j3": 64})
    sch.vectorize(["j3"])
    sch.buffer_at("j")
    sch.buffer_at("j1")
    print(sch)
    # Expected in TVM schedule
    return [
        "reorder(i, j, i1, j1, k, i2, j2, k1, i3, j3)",
        "reverse_compute_at(O_W0, j)",
        "reverse_compute_at(O_W0, j1)",
    ]

def sched_tile_unroll_vec(sch):
    sch.tile("i", {"i1": 8})
    sch.tile("j", {"j1": 48})
    sch.tile("k", {"k1": 50})
    sch.interchange(["j", "k", "i", "k1", "i1", "j1"])
    sch.parallelize(["j"])
    sch.unroll({"k1": 32, "i1": 8})
    sch.vectorize(["j1"])
    print(sch)
    # Expected in TVM schedule
    return [
        "reorder(j, k, i, k1, __u_k1, i1, j1, __v_j1)",
        "unroll(__u_k1)",
        "unroll(i1)",
        "unroll(j1)",
        "vectorize(__v_j1)",
    ]

def check_schedule(impl, sched_func):
    sch = impl.get_scheduler()
    expected = sched_func(sch)
    schedule = sch.schedule()
    schedule_str = str(schedule)
    print(f"TVM schedule:\n{schedule_str}")
    for substr in expected:
        assert substr in schedule_str
    return schedule

def check_evaluate(impl, schedule):
    result = impl.evaluate(schedule)
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

@requires_tvm
def test_sched_nop():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile2():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile2p():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2p)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile3wc():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile3wc)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile_unroll_vec():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile_unroll_vec)
    check_evaluate(impl, schedule)

def check_compile_evaluate(imp, schedule, compiler_args, evaluate_args):
    compiler = imp.get_compiler(
        **compiler_args,
    )
    module = compiler.compile(schedule)
    evaluator = module.get_evaluator(
        **evaluate_args,
    )
    results, code, error_msg = evaluator.evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"
    print(f"Results: {results}")
    assert isinstance(results[0], float) and float(results[0]) > 0

@requires_tvm
@pytest.mark.parametrize(
    "module_type",
    (
        "shared_lib",
        "emit_c",
        "ar_lib",
    ),
)
@pytest.mark.parametrize(
    "bare_ptr",
    (
        False,
        True,
    ),
)
def test_backend_variant(tmpdir, module_type, bare_ptr):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    libpath = Path(tmpdir) / impl.graph.name
    schedule = check_schedule(impl, sched_tile2p)
    check_compile_evaluate(
        impl,
        schedule,
        {
            "dump_file": str(libpath),
            module_type: True,
            "bare_ptr": bare_ptr,
        },
        {
            "validate": True,
        }
    )
