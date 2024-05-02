from abc import ABC
import subprocess
import numpy

from mlir.ir import *
import mlir
from mlir.dialects import (
    arith,
    builtin,
    func,
    linalg,
    tensor
)

from PerfectlyNestedImplementer import PerfectlyNestedImplementer
import transform

class MMImplementer(PerfectlyNestedImplementer):

    def __init__(
            self,
            mlir_install_dir: str,
            dims: dict[str,int],
            parallel_dims: list[str],
            reduction_dims: list[str]
    ):
        super().__init__(mlir_install_dir,dims,parallel_dims,reduction_dims)

    def payload(self,m,elt_type):
        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        k = self.dims[self.reduction_dims[0]]
        A_tensor_type = RankedTensorType.get((i,k), elt_type)
        B_tensor_type = RankedTensorType.get((k,j), elt_type)
        C_tensor_type = RankedTensorType.get((i,j), elt_type)
        with InsertionPoint.at_block_begin(m.body):
            f = func.FuncOp(
                name=self.payload_name,
                type=FunctionType.get(
                    inputs=[A_tensor_type, B_tensor_type],
                    results=[C_tensor_type]
                ),
            )
        with InsertionPoint(f.add_entry_block()):
            A = f.entry_block.arguments[0]
            B = f.entry_block.arguments[1]
            zero = arith.ConstantOp(
                value=FloatAttr.get(elt_type, 0.0), result=elt_type
            ).result
            C_init = tensor.SplatOp(C_tensor_type,zero)
            matmul = linalg.matmul(A, B, outs=[C_init])
            func.ReturnOp([matmul])
        return f
        
    def uniquely_match(self):

        dims = self.dims.values()
        
        sym_name,input_var,seq_sig = transform.get_seq_signature(
            input_consumed = False,
            has_output = True,
        )
    
        res_var, global_match_sig = transform.get_match_sig(input_var)
        bb_input_var,bb_header = transform.get_bb_header()
    
        match_dims = transform.get_match_dims(bb_input_var,dims)
        
        match_opname = transform.get_match_op_name(bb_input_var,"linalg.matmul")

        tmyield = transform.get_match_structured_terminator(bb_input_var)
    
        tyield = transform.get_terminator(result = res_var)
    
        lines = [
            seq_sig,
            "{",
            global_match_sig,
            "{",
            bb_header,
        ] + match_dims + [
            match_opname,
            tmyield,
            "}",
            tyield,
            "}"
        ]

        return sym_name,"\n".join(lines)

    def main(self,m,frtclock,fprint,fmatmul,elt_type):
        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        k = self.dims[self.reduction_dims[0]]
        A_tensor_type = RankedTensorType.get((i,k), elt_type)
        B_tensor_type = RankedTensorType.get((k,j), elt_type)
        with InsertionPoint.at_block_begin(m.body):
            fmain = func.FuncOp(
                name="main",
                type=FunctionType.get(inputs=[], results=[]),
        )
        with InsertionPoint(fmain.add_entry_block()):
            #
            rand1 = arith.ConstantOp(
                value=FloatAttr.get(elt_type, numpy.random.random()), result=elt_type
            ).result
            A = tensor.SplatOp(A_tensor_type,rand1)
            #
            rand2 = arith.ConstantOp(
                value=FloatAttr.get(elt_type, numpy.random.random()), result=elt_type
            ).result
            B = tensor.SplatOp(B_tensor_type,rand2)
            #
            callrtclock1 = func.CallOp(frtclock,[])
            C = func.CallOp(fmatmul, [A,B])
            callrtclock2 = func.CallOp(frtclock,[])
            time = arith.SubFOp(callrtclock2,callrtclock1)
            func.CallOp(fprint,[time])
            func.ReturnOp([])
        
        return fmain
