import transform
from mlir.ir import *
from mlir.dialects import (
    builtin,
    func,
)

def pack_schedules(mlir_impls,sym_name):
    myvar = transform.get_new_var()
    sym_name,input_var,seq_sig = transform.get_seq_signature(
        input_var = myvar,
        sym_name=sym_name
    )
    schedules = []
    for impl in mlir_impls:
        schedules += impl.materialize_schedule(input_var=myvar)
    integrated_schedules = (
        [seq_sig,"{"]
        + schedules
        + [transform.get_terminator(),"}"]
    )
    trans_script = (
        "module attributes {transform.with_named_sequence} {"
        + "\n"
        + "\n".join(integrated_schedules)
        + "\n"
        + "}"
    )
    return trans_script

def integrate(mlir_impls,module,ctx):
    module.operation.attributes["transform.with_named_sequence"] = UnitAttr.get(context=ctx)
    trans_script =  pack_schedules(
        mlir_impls=mlir_impls,
        sym_name="@__transform_main"
    )
    trans_match = Module.parse(trans_script,context=ctx)
    with InsertionPoint(module.body):
        for o in trans_match.body.operations:
            o.operation.clone()
    
def insert_rtclock(module,ctx,loc):
    f64 = F64Type.get(context = ctx)
    with InsertionPoint.at_block_begin(module.body):
        frtclock = func.FuncOp(
            name="rtclock",
            type=FunctionType.get(
                inputs=[], results=[f64]
            ),
            visibility="private",
            loc = loc,
        )
    return frtclock

def insert_printF64(module,ctx,loc):
    f64 = F64Type.get(context = ctx)
    with InsertionPoint.at_block_begin(module.body):
        fprint = func.FuncOp(
            name="printF64",
            type=FunctionType.get(
                inputs=[f64], results=[]
            ),
            visibility="private",
            loc = loc,
        )
    return fprint
