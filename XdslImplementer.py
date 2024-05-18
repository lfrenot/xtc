from abc import ABC
import subprocess

from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.ir import Block, Region, MLContext, Operation
from xdsl.dialects.builtin import (
    ModuleOp,
    DenseIntOrFPElementsAttr,
    TensorType,
    MemRefType,
    f32,
    f64,
)
from xdsl.dialects import func,arith,linalg
from AbsImplementer import AbsImplementer
from PerfectlyNestedImplementer import PerfectlyNestedImplementer
import transform

class XdslImplementer(PerfectlyNestedImplementer):

    def __init__(
            self,
            mlir_install_dir: str,
            source_op: Operation,
            dims: dict[str,int],
            parallel_dims: list[str],
            reduction_dims: list[str],
    ):
        #
        super().__init__(
            mlir_install_dir,
            source_op,
            dims,
            parallel_dims,
            reduction_dims
        )

    def payload(self):
        # Fetch data
        operands = self.source_op.operands
        inputs = self.source_op.inputs
        inputs_types = [o.type for o in inputs]
        results_types = [r.type for r in self.source_op.results]
        #
        payload = Block(arg_types=inputs_types)
        outputs = self.outputs_init()
        outputs_vars = []
        for o in outputs:
            outputs_vars += o.results
        concrete_operands = list(payload.args) + outputs_vars
        value_mapper = {o:p for o,p in zip(operands,concrete_operands)}
        
        new_op = self.source_op.clone(
            value_mapper = value_mapper
        )
        payload.add_ops(outputs + [new_op,func.Return(new_op)])
        payload_func = func.FuncOp.from_region(
            self.payload_name,
            inputs_types,
            results_types,
            Region(payload)
        )
        return payload_func

    def inputs_init(self):
        inputs_types = [o.type for o in self.source_op.inputs]
        return [
            arith.Constant(DenseIntOrFPElementsAttr.tensor_from_list(
                [1.0],
                ty.get_element_type(),
                ty.get_shape()
            )) for ty in inputs_types
        ]
    

    def outputs_init(self):
        outputs_types = [o.type for o in self.source_op.outputs]
        return [
            arith.Constant(DenseIntOrFPElementsAttr.tensor_from_list(
                [0.0],
                ty.get_element_type(),
                ty.get_shape()
            )) for ty in outputs_types
        ]

    def main(self,ext_rtclock,ext_printF64,payload_func):
        results_types = [r.type for r in self.source_op.results]
        #
        inputs = self.inputs_init()
        rtclock_call1 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        outputs = self.outputs_init()
        payload_call = func.Call(
            payload_func.sym_name.data,
            inputs,
            results_types
        )
        rtclock_call2 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        elapsed = arith.Subf(rtclock_call2,rtclock_call1)
        print_elapsed = func.Call(ext_printF64.sym_name.data, [elapsed], [])
        main = Block()
        main.add_ops(inputs + [
            rtclock_call1,
            payload_call,
            rtclock_call2,
            elapsed,
            print_elapsed,
            func.Return(),
        ])
        main_func = func.FuncOp.from_region("main", [], [], Region(main))
        return main_func

    def build_rtclock(self):
        return func.FuncOp.external("rtclock", [], [f64])

    def build_printF64(self):
        return func.FuncOp.external("printF64", [f64], [])
