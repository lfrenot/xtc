import copy
from collections import OrderedDict

sym_count = 0
var_count = 0

def get_new_var():
    global var_count
    new_var = f"%{var_count}"
    var_count += 1
    return new_var

def get_new_seq_name():
    global sym_count
    new_seq_name = f"@seq{sym_count}"
    sym_count += 1
    return new_seq_name

def get_seq_signature(
        input_consumed = False,
        has_output = False
):
    sym_name = get_new_seq_name()
    input_var = get_new_var()

    input_attribute = ""
    if input_consumed:
        input_attribute = "{transform.consumed}"
    else:
        input_attribute = "{transform.readonly}"

    tail = " -> (!transform.any_op)" if has_output else ""

    seq_sig = (
        f"transform.named_sequence {sym_name} ({input_var}: "
        + f"!transform.any_op {input_attribute}) {tail}"
    )

    return sym_name, input_var, seq_sig

def get_match_dims(
        input_var,
        dims,
):
    dims_var = get_new_var()
    dims_var_list = [get_new_var() for d in dims]
    merge_dims_var = get_new_var()
    get_dims = (
        f"{dims_var} = transform.match.structured.dim {input_var}[all]:"
        + "(!transform.any_op) -> !transform.param<i64>"
    )
    init_dims = [
        f"{vd} = transform.param.constant {d} : i64 -> !transform.param<i64>"
        for vd,d in zip(dims_var_list,dims)
    ]
    merge_dims = (
        f"{merge_dims_var} = transform.merge_handles "
        + ", ".join(dims_var_list)
        + ": !transform.param<i64>"
    )
    cmp_dims = (
        f"transform.match.param.cmpi eq {dims_var}, {merge_dims_var}"
        + ": !transform.param<i64>"
    )
    return [get_dims] + init_dims + [merge_dims,cmp_dims]
    
def get_match_op_name(input_var,name):
    return (
        f"transform.match.operation_name {input_var}"
        + "[\"" + name + "\"]: !transform.any_op"
    )

def get_terminator(namespace = "transform", result = None):
    tail = f" {result} : !transform.any_op" if result else ""
    return f"{namespace}.yield{tail}"

def get_match_structured_terminator(result = None):
    return get_terminator(namespace = "transform.match.structured", result=result)

def get_match_sig(input_var):
    res_var = get_new_var()
    global_match_sig = (
        f"{res_var} = transform.match.structured failures(propagate) "
        + f"{input_var} : (!transform.any_op) -> !transform.any_op"
    )
    return res_var, global_match_sig

def get_bb_header():
    bb_input_var = get_new_var()
    bb_header = f"^bb0({bb_input_var}: !transform.any_op):"
    return bb_input_var, bb_header

def get_vectorize(op):
    return f"transform.structured.vectorize {op} : !transform.any_op"

def get_unroll(loop,factor):
    return (
        f"transform.loop.unroll {loop}"
        + "{ factor = "
        + str(factor)
        + "} : !transform.any_op"
    )

def produce_tiling_instr(
            current_state,
            dims_vector,
    ):

    new_state = get_new_var()
    new_loop = get_new_var()
    
    str_dims = "[" + ",".join([str(d) for d in dims_vector]) + "]"

    str_tile = (
        f"{new_state},{new_loop} = transform.structured.tile_using_for"
        + f"{current_state} {str_dims} :"
        + "(!transform.any_op) -> (!transform.any_op, !transform.any_op)"
    )
    return new_state,new_loop,str_tile

def build_main(matchers_transformers):
    sym_name = "@__transform_main"

    input_var = get_new_var()
    bufferized_var = get_new_var()

    seq_sig = (
        f"transform.named_sequence {sym_name} ({input_var}: "
        + "!transform.any_op {transform.consumed})"
    )

    bufferization = (
        f"{bufferized_var} = transform.bufferization.one_shot_bufferize "
        + f"{input_var} : (!transform.any_op) -> !transform.any_op"
    )

    branches = []
    current_state = bufferized_var
    for (matcher,transformer) in matchers_transformers:
        new_state = get_new_var()
        shot = (
            f"{new_state} = transform.foreach_match in {current_state} "
            + f"{matcher} -> {transformer} : (!transform.any_op) -> !transform.any_op"
        )
        branches.append(shot)
        current_state = new_state

    tyield = "transform.yield"
        
    lines = [
        seq_sig,
        "{",
        bufferization,
    ] + branches + [
        tyield,
        "}"
    ]
    return sym_name,"\n".join(lines)
