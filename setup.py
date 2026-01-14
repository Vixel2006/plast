import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class custom_build_ext(build_ext):
    def build_extension(self, ext):
        # This is a bit of a hacky setup.py for CUDA without Pytorch.
        # We will manually compile .cu files with nvcc and .c files with cc,
        # then link them with pybind11.
        
        include_dirs = [
            "include",
            "include/kernels/cuda",
            "include/optimizers",
            "/usr/local/cuda/include",
            pybind11.get_include()
        ]
        
        c_sources = []
        cu_sources = []
        for src in ext.sources:
            if src.endswith(".c") or src.endswith(".cpp"):
                c_sources.append(src)
            elif src.endswith(".cu"):
                cu_sources.append(src)
        
        objs = []
        
        # Compile CU sources
        for src in cu_sources:
            obj = src + ".o"
            cmd = ["nvcc", "-O3", "-arch=sm_80", "-Xcompiler", "-fPIC", "-DCUDA_AVAILABLE"]
            for d in include_dirs:
                cmd.extend(["-I", d])
            cmd.extend(["-c", src, "-o", obj])
            print(f"Compiling CUDA: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            objs.append(obj)
            
        # Compile C/CPP sources
        python_includes = subprocess.check_output(["python3-config", "--includes"]).decode().split()
        
        for src in c_sources:
            obj = src + ".o"
            compiler = "cc"
            if src.endswith(".cpp"):
                compiler = "g++"
            
            cmd = [compiler, "-O3", "-fPIC", "-march=native", "-fopenmp", "-DCUDA_AVAILABLE"]
            for d in include_dirs:
                cmd.extend(["-I", d])
            cmd.extend(python_includes)
            cmd.extend(["-c", src, "-o", obj])
            print(f"Compiling {src}: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            objs.append(obj)
            
        # Link everything into a shared library
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # pybind11 linking flags
        ld_flags = subprocess.check_output(["python3-config", "--ldflags"]).decode().split()
        # Remove some flags that might cause issues in shared libs if needed
        
        link_cmd = ["nvcc", "-shared", "-o", ext_path] + objs
        link_cmd.extend(["-L/usr/local/cuda/lib64", "-lcudart", "-lgomp"])
        # Add pybind11 and python flags
        python_includes = subprocess.check_output(["python3-config", "--includes"]).decode().split()
        link_cmd.extend(python_includes)
        link_cmd.extend(["-I" + pybind11.get_include()])
        
        print(f"Linking: {' '.join(link_cmd)}")
        subprocess.check_call(link_cmd)

# Gather all source files
c_sources = [
    "src/arena.c", "src/arena_cpu.c", "src/graph.c", "src/node.c", "src/op.c", "src/tensor.c",
    "src/kernels/cpu/abs.c", "src/kernels/cpu/add.c", "src/kernels/cpu/broadcast.c",
    "src/kernels/cpu/conv2d.c", "src/kernels/cpu/cos.c", "src/kernels/cpu/cpu_tensor_init.c",
    "src/kernels/cpu/cpu_utils.c", "src/kernels/cpu/div.c", "src/kernels/cpu/expand.c",
    "src/kernels/cpu/exp.c", "src/kernels/cpu/flatten.c", "src/kernels/cpu/leaky_relu.c",
    "src/kernels/cpu/log.c", "src/kernels/cpu/matmul.c", "src/kernels/cpu/max.c",
    "src/kernels/cpu/mean.c", "src/kernels/cpu/min.c", "src/kernels/cpu/mul.c",
    "src/kernels/cpu/neg.c", "src/kernels/cpu/sin.c", "src/kernels/cpu/squeeze.c",
    "src/kernels/cpu/sub.c", "src/kernels/cpu/sum.c", "src/kernels/cpu/tan.c",
    "src/kernels/cpu/transpose.c", "src/kernels/cpu/unsqueeze.c", "src/kernels/cpu/view.c",
    "src/optimizers/cpu/adam.c", "src/optimizers/cpu/adamw.c", "src/optimizers/cpu/sgd.c",
    "src/optimizers/cpu/zero_grad.c",
    "src/python/bindings.cpp"
]

cu_sources = [
    "src/arena_cuda.cu", "src/kernels/cuda/abs.cu", "src/kernels/cuda/add.cu",
    "src/kernels/cuda/conv2d.cu", "src/kernels/cuda/cos.cu", "src/kernels/cuda/cuda_tensor_init.cu",
    "src/kernels/cuda/div.cu", "src/kernels/cuda/exp.cu", "src/kernels/cuda/leaky_relu.cu",
    "src/kernels/cuda/log.cu", "src/kernels/cuda/matmul.cu", "src/kernels/cuda/max.cu",
    "src/kernels/cuda/mean.cu", "src/kernels/cuda/min.cu", "src/kernels/cuda/mul.cu",
    "src/kernels/cuda/neg.cu", "src/kernels/cuda/sin.cu", "src/kernels/cuda/sub.cu",
    "src/kernels/cuda/sum.cu", "src/kernels/cuda/tan.cu", "src/optimizers/cuda/adam.cu",
    "src/optimizers/cuda/adamw.cu", "src/optimizers/cuda/sgd.cu", "src/optimizers/cuda/zero_grad.cu"
]

setup(
    name="plast",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "plast.plast_core",
            sources=c_sources + cu_sources,
        )
    ],
    cmdclass={"build_ext": custom_build_ext},
    install_requires=["numpy"],
)
