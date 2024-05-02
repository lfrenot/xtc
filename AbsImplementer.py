from abc import ABC
from abc import abstractmethod
import subprocess

import xdsl
from xdsl.dialects.builtin import (
    f64,
)

from mlir.ir import *
from mlir.dialects import (
    builtin,
    func
)

import transform

transform_opt = "transform-interpreter"
transform_opts = [
    f"--{transform_opt}",
]

lowering_opts = [
    "--func-bufferize",
    # "--buffer-deallocation",
    "--test-transform-dialect-erase-schedule",
    "--convert-scf-to-cf",
    "--canonicalize",
    "--convert-vector-to-llvm=enable-x86vector",
    "--test-lower-to-llvm",
]

mliropt_opts = transform_opts +  lowering_opts

obj_dump_file = "/tmp/dump.o"

mlirrunner_opts = [
    "-e",
    "main",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = 'objdump'

objdump_opts = [
    '-d',
    '--no-addresses',
    '--no-show-raw-insn',
    '--visualize-jumps'
]

objdump_color_opts = [
    '--visualize-jumps=color',
    '--disassembler-color=on',
]

def build_rtclock(m):
    f64 = F64Type.get()
    with InsertionPoint.at_block_begin(m.body):
        frtclock = func.FuncOp(
            name="rtclock",
            type=FunctionType.get(
                inputs=[], results=[f64]
            ),
            visibility="private",
        )
    return frtclock

def build_printF64(m):
    f64 = F64Type.get()
    with InsertionPoint.at_block_begin(m.body):
        fprint = func.FuncOp(
            name="printF64",
            type=FunctionType.get(
                inputs=[f64], results=[]
            ),
            visibility="private",
        )
    return fprint

class AbsImplementer(ABC):

    count = 0

    def __init__(
            self,
            mlir_install_dir: str,
    ):
               #
        self.payload_name = f"payload{AbsImplementer.count}"
        AbsImplementer.count += 1
        #
        self.mliropt = f"{mlir_install_dir}/bin/mlir-opt"
        self.cmd_run_mlir = [
            f"{mlir_install_dir}/bin/mlir-cpu-runner",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_runner_utils.so",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_c_runner_utils.so"  
        ] + mlirrunner_opts
        self.cmd_disassembler = (
            [objdump_bin]
            + objdump_opts
            + [f'--disassemble={self.payload_name}']
        )

    def build_compile_extra_opts(
            self,
            print_source_ir,
            print_transformed_ir,
            print_ir_after,
            print_ir_before,
            color,
    ):
        compile_extra_opts = []
        if print_source_ir:
            zero_opt = mliropt_opts[0].replace('--','')
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_opt}")
        if print_transformed_ir:
            zero_lowering_opt = lowering_opts[0].replace('--','')
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_lowering_opt}")
        compile_extra_opts += [f"--mlir-print-ir-after={p}" for p in print_ir_after]
        compile_extra_opts += [f"--mlir-print-ir-before={p}" for p in print_ir_before]
        return compile_extra_opts

    def build_disassemble_extra_opts(
            self,
            print_assembly,
            color
    ):
        disassemble_extra_opts = []
        if print_assembly:
            disassemble_extra_opts += [obj_dump_file]
        if color:
            disassemble_extra_opts += objdump_color_opts
        return disassemble_extra_opts

    def build_run_extra_opts(
            self,
            print_assembly,
            color
    ):
        run_extra_opts = []
        if print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={obj_dump_file}",
            ]
        return run_extra_opts
    
    def evaluate(
            self,
            print_source_ir = False,
            print_transformed_ir = False,
            print_ir_after = [],
            print_ir_before = [],
            print_assembly = False,
            color = True,
    ):

        str_module = self.glue()

        run_extra_opts = []

        compile_extra_opts = self.build_compile_extra_opts(
            print_source_ir = print_source_ir,
            print_transformed_ir = print_transformed_ir,
            print_ir_after = print_ir_after,
            print_ir_before = print_ir_before,
            color = color,
        )

        disassemble_extra_opts = self.build_disassemble_extra_opts(
            print_assembly = print_assembly,
            color = color
        )

        run_extra_opts = self.build_run_extra_opts(
            print_assembly = print_assembly,
            color = color
        )
            
        module_llvm = subprocess.run(
            [self.mliropt] + mliropt_opts + compile_extra_opts,
            input=str_module,
            stdout=subprocess.PIPE,
            text=True
        )
        result = subprocess.run(
            self.cmd_run_mlir + run_extra_opts,
            input=module_llvm.stdout,
            stdout=subprocess.PIPE,
            text=True
        )
        if print_assembly:
            subprocess.run(
                self.cmd_disassembler + disassemble_extra_opts,
                text=True
            )
            
        return result.stdout

    def glue(self):
        ctx = Context()
        with Location.unknown(ctx) as loc:
            elt_type = F32Type.get()
            m = builtin.ModuleOp()
            ext_rtclock = build_rtclock(m)
            ext_printF64 = build_printF64(m)
            payload_func = self.payload(m,elt_type)
            main_func = self.main(m,ext_rtclock,ext_printF64,payload_func,elt_type)
        # Glue the module
        # mod = ModuleOp([ext_rtclock,ext_printF64,payload_func,main_func])
        # str_mod = str(mod)
        str_mod = "\n".join([str(tl) for tl in [
            ext_rtclock,
            ext_printF64,
            payload_func,
            main_func,
        ]])
        match_sym_name,str_trans_match = self.uniquely_match()
        
        sched_sym_name,str_trans_sched = self.materialize_schedule()
        
        main_name,str_trans_main = transform.build_main(
            [(match_sym_name,sched_sym_name)]
        )
        str_glued = (
            "module attributes {transform.with_named_sequence} {"
            + "\n"
            + str_mod
            + "\n"
            + str_trans_sched
            + "\n"
            + str_trans_match
            + "\n"
            + str_trans_main
            + "\n"
            + "}"
        )
        
        return str_glued
    
    @abstractmethod
    def payload(self,m,elt_type):
        pass

    @abstractmethod
    def uniquely_match(self):
        pass

    @abstractmethod
    def materialize_schedule(self):
        pass

    @abstractmethod
    def main(self):
        pass
