import sys
from abc import ABC
import subprocess
import numpy

import mlir
from mlir.ir import (
    Module,
    InsertionPoint,
    Context,
    Location,
    IntegerType,
    F32Type,
    F64Type,
    MemRefType,
    FunctionType,
    UnitAttr
)

from mlir.dialects import (
    arith,
    builtin,
    func,
    linalg,
    tensor,
    bufferization,
    memref
)
from xdsl.utils.hints import isa
from xdsl.ir import Operation
from xdsl.dialects.builtin import f32,IntegerType as xdslIntegerType
from xdsl.dialects.builtin import ArrayAttr as xdslArrayAttr
from xdsl.dialects.builtin import DictionaryAttr as xdslDictionaryAttr
from xdsl.dialects.builtin import UnitAttr as xdslUnitAttr
from xdsl.dialects.builtin import FloatAttr as xdslFloatAttr
from xdsl.dialects.builtin import AnyIntegerAttr as xdslAnyIntegerAttr
from xdsl.dialects.builtin import AnyShapedType as xdslAnyShapedType
from xdsl.dialects.builtin import AnyMemRefType as xdslAnyMemRefType
from xdsl.dialects.arith import Constant as xdslConstant

from xdsl.ir import Block as xdslBlock
from xdsl.ir import Region as xdslRegion
from xdsl.dialects import func as xdslfunc

from AbsImplementer import AbsImplementer
import transform
import mlir_packing

class MlirImplementer(AbsImplementer):

    count = 0
    
    def __init__(
            self,
            mlir_install_dir: str,
            source_op: Operation,
            dims: dict[str,int],
            parallel_dims: list[str],
            reduction_dims: list[str],
            vectors_size = 16,
            payload_name = None,
    ):

        super().__init__(mlir_install_dir,vectors_size)
        #
        self.ext_rtclock = mlir_packing.insert_rtclock(self.module,self.ctx,self.loc)
        self.ext_printF64 = mlir_packing.insert_printF64(self.module,self.ctx,self.loc)
        #
        self.payload_name = payload_name if payload_name else f"payload{MlirImplementer.count}"
        self.init_payload_name = f"init_{self.payload_name}"
        self.op_id_attribute = f"id{MlirImplementer.count}"
        MlirImplementer.count += 1
        #
        self.source_op = source_op
        self.source_op.attributes[self.op_id_attribute] = xdslUnitAttr()

        # Parse shaped (no scalar) inputs and outputs
        self.inputs_types = []
        for i in self.source_op.inputs:
            if isa(i.type,xdslAnyMemRefType):
                mlirI = MemRefType.parse(str(i.type),context=self.ctx)
                self.inputs_types.append(mlirI)
        self.outputs_types = []
        for o in self.source_op.outputs:
            if isa(o.type,xdslAnyMemRefType):
                mlirO = MemRefType.parse(str(o.type),context=self.ctx)
                self.outputs_types.append(mlirO)
        # Generate the payload
        xdsl_func = self.xdsl_operator_to_function()
        payload_func = func.FuncOp.parse(str(xdsl_func),context = self.ctx)
        ip = InsertionPoint.at_block_begin(self.module.body)
        ip.insert(payload_func)
        self.main(payload_func)
        #
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.reduction_dims = reduction_dims
        self.tiles = {k:{k:1} for k,v in self.dims.items()}
        self.tile_sizes_cache = {}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling = dict([])
        #

    def loops(self):
        loops = dict()
        for tile_level in range(len(max(self.tiles.values(),key=len))):
            for (k,v) in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
        return loops
    
    def get_default_interchange(self):
        return list(self.loops().keys())

    def tile(
            self,
            dim: str,
            tiles: dict[str,int],
    ):
        #
        for k,v in tiles.items():
            self.tile_sizes_cache[k] = v
        #
        ndims = list(tiles.keys())
        tiles_sizes = list(tiles.values())
        
        assert(len(ndims) == len(tiles_sizes))

        previous_tile_size = self.dims[dim]
        for ts in tiles_sizes:
            assert(previous_tile_size%ts == 0)
            previous_tile_size = ts
        
        dims = [dim] + ndims
        sizes = tiles_sizes + [1]
        for d,s in zip(dims,sizes):
            self.tiles[dim][d] = s
        self.permutation = self.get_default_interchange()

        if dim in self.parallel_dims:
            self.parallel_dims += ndims
        if dim in self.reduction_dims:
            self.reduction_dims += ndims
            
    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def vectorize(self, vectorization: list[str]):
        self.vectorization = vectorization
        self.propagate_vectorization()

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert(p in self.parallel_dims)
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str,int]):
        self.unrolling = unrolling
        self.propagate_vectorization()

    def propagate_vectorization(self):
        nvect = []
        for v in self.vectorization:
            ind = self.permutation.index(v)
            for i in list((range(0,ind)))[::-1]:
                dim = self.permutation[i]
                if dim in self.unrolling and dim in self.parallel_dims:
                    nvect.append(dim)
                else:
                    break
        for nv in nvect:
            self.vectorization.append(nv)
            self.unrolling.pop(nv)
        
    

    def xdsl_operator_to_function(self):
        # Fetch data
        operands = self.source_op.operands
        shaped_types,scalar_types = [],[]
        for o in operands:
            if isa(o.type,xdslAnyMemRefType):
                shaped_types.append(o.type)
            else:
                scalar_types.append(o.type)
            
        #
        payload = xdslBlock(arg_types=shaped_types)
        concrete_operands = []
        shaped_count,scalar_count = 0,0
        for o in operands:
            if isa(o.type,xdslAnyMemRefType):
                concrete_operands.append(payload.args[shaped_count])
                shaped_count += 1
            else:
                if isa(o.type,xdslIntegerType):
                    attr = xdslAnyIntegerAttr(0,scalar_types[scalar_count])
                else:
                    attr = xdslFloatAttr(0.0,scalar_types[scalar_count])
                constant = xdslConstant(attr)
                payload.add_ops([constant])
                concrete_operands.append(constant.results[0])
                scalar_count += 1
                
        value_mapper = {o:p for o,p in zip(operands,concrete_operands)}
        
        new_op = self.source_op.clone(
            value_mapper = value_mapper
        )
        payload.add_ops([new_op,xdslfunc.Return()])
        payload_func = xdslfunc.FuncOp(
            name=self.payload_name,
            function_type=(shaped_types, ()),
            region=xdslRegion(payload),
            arg_attrs=xdslArrayAttr(param=[
                xdslDictionaryAttr(data={ 'llvm.noalias': xdslUnitAttr() })
                for t in shaped_types
            ]),
        )
        
        return payload_func
    
    def main(self,fmatmul):
        #
        with InsertionPoint.at_block_begin(self.module.body):
            fmain = func.FuncOp(
                name="entry",
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
        with InsertionPoint(fmain.add_entry_block()), self.loc as loc:
            inputs = []
            for ity in self.inputs_types:
                if IntegerType.isinstance(ity.element_type):
                    v = int(numpy.random.random())
                else:
                    v = numpy.random.random()
                scal = arith.ConstantOp(ity.element_type,v)
                mem = memref.AllocOp(ity, [], [])
                linalg.fill(scal,outs=[mem])
                inputs.append(mem)
            #
            callrtclock1 = func.CallOp(self.ext_rtclock,[],loc=self.loc)

            for oty in self.outputs_types:
                v = 0 if IntegerType.isinstance(oty.element_type) else 0.0
                scal = arith.ConstantOp(oty.element_type,v)
                mem = memref.AllocOp(oty, [], [])
                linalg.fill(scal,outs=[mem])
                inputs.append(mem)
            
            func.CallOp(fmatmul, inputs,loc=self.loc)
            callrtclock2 = func.CallOp(self.ext_rtclock,[],loc=self.loc)
            
            time = arith.SubFOp(callrtclock2,callrtclock1,loc=self.loc)
            func.CallOp(self.ext_printF64,[time],loc=self.loc)

            for i in inputs:
                memref.DeallocOp(i)
            
            func.ReturnOp([],loc=self.loc)
        
        return fmain

    def integrate(self):
        assert(not self.integrated)
        mlir_packing.integrate([self],self.module,self.ctx)
        self.integrated = True

    def materialize_tiling(self,global_handle):

        loop_to_tile,match_attr = transform.match_by_attribute(
            global_handle,
            self.op_id_attribute
        )
        
        # Build the transform vectors corresponding to each tiling instruction
        positions = {}
        dims_vectors = {}
        for tile_level in range(len(max(self.tiles.values(),key=len))):
            for dim_to_tile,(k,v) in enumerate(self.tiles.items()):
                if tile_level >= len(v):
                    continue
                #
                dim_name = list(v.keys())[tile_level]
                dims_vectors[dim_name] = [
                    v[dim_name] if i == dim_to_tile else 0
                    for i in range(len(self.tiles))
                ]
                positions[dim_name] = dim_to_tile

        # Reorder the vectors (according to the permutation)
        dims_vectors = dict([(p,dims_vectors[p]) for p in self.permutation])

        # Actually produce the tiling instructions and annotate the resulting
        # loops
        current_state = loop_to_tile
        tiling_instrs = [match_attr]
        loops = []
        for dim,dims_vector in dims_vectors.items():
            # Useless to materialize a loop which will be vectorized
            if dim in self.vectorization:
            # if self.vectorization_backpropagation_possible(dim):
                break
            # The actual tiling
            current_state,new_loop,new_instr = transform.produce_tiling_instr(
                current_state = current_state,
                dims_vector = dims_vector,
                parallel = dim in self.parallelization,
            )
            loops.append(new_loop)
            # Name the resulting loop
            annot = transform.annotate(new_loop,f"{self.op_id_attribute}_{dim}")
            #
            tiling_instrs += [new_instr + annot]

        return tiling_instrs,loops[0]

    def normalize_and_vectorize(self,tiled_loop):
        parent, parent_instr = transform.get_parent(tiled_loop)

        # Canonicalize the code produced by the tiling operations
        norm_instrs = transform.tiling_apply_patterns(parent)

        # Produce the vectorization instructions
        vect_instrs = []
        vectorized = parent
        if len(self.vectorization) > 0:
            handler,vectorize = transform.get_vectorize_children(parent)
            pre_hoist = transform.vector_pre_hoist_apply_patterns(handler)
            hoisted0,get_hoist0 = transform.vector_hoist(handler)
            vect_instrs = [vectorize] + pre_hoist + [get_hoist0]
            vectorized = hoisted0
        else:
            vectorized, scalarization = transform.get_scalarize(current_state)
            vect_instrs = [scalarization]

        return [parent_instr] + norm_instrs + vect_instrs, vectorized

    def materialize_unrolling(self,vectorized):
        last_handler = vectorized
        # Produce the unrolling instructions using the annotations on loops
        unroll_instrs = []
        for dim,factor in self.unrolling.items():
            # loop,match_loop = transform.match_by_attribute(vectorized,dim)
            loop,match_loop = transform.match_by_attribute(vectorized,f"{self.op_id_attribute}_{dim}")
            unroll = transform.get_unroll(loop,factor)
            unroll_instrs += [match_loop,unroll]
            last_handler = loop

        postprocess = []
        if len(self.vectorization) > 0:
            hoisted,get_hoist = transform.vector_hoist(vectorized)
            get_lower = transform.vector_lower_outerproduct_patterns(hoisted)
            postprocess = [get_hoist] + get_lower
            last_handler = hoisted

        # if len(unroll_instrs) > 0:
        if False:
            continuation, parent_instr = transform.get_parent(last_handler)
            postprocess.append(parent_instr)
        else:
            continuation = last_handler
            
        return unroll_instrs + postprocess, continuation
    
    def materialize_schedule(self,input_var):
            
        tiling_instrs,tiled_loop = self.materialize_tiling(global_handle=input_var)

        vect_instrs,vectorized = self.normalize_and_vectorize(tiled_loop)
        
        unroll_instrs,unrolled = self.materialize_unrolling(vectorized)
        
        return tiling_instrs + vect_instrs + unroll_instrs
        
    @classmethod
    def _np_types_spec(cls, types):
        types_map = {"f32": "float32", "f64": "float64"}
        types_spec = [
            {
                "shape": t.get_shape(),
                "dtype": types_map[str(t.get_element_type())],
            }
            for t in types]
        return types_spec

    def np_inputs_spec(self):
        return self._np_types_spec([i.type for i in self.source_op.inputs])

    def np_outputs_spec(self):
        return self._np_types_spec([i.type for i in self.source_op.outputs])

    def reference_impl(self, *operands):
        if self.source_op.name == "linalg.matmul":
            np.matmul(operands[0], operands[1], out=operands[2])
        else:
            assert 0, f"unknown implementation for operation: {self.source_op.name}"
