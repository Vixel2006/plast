import os, sys, subprocess, shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11

HAS_CUDA = shutil.which("nvcc") is not None

class custom_build_ext(build_ext):
    def build_extension(self, ext):
        include_dirs = [
            "include", "include/kernels/cuda", "include/optimizers",
            pybind11.get_include(),
        ]
        cuda_include = "/usr/local/cuda/include"
        if HAS_CUDA and os.path.isdir(cuda_include):
            include_dirs.insert(0, cuda_include)

        c_srcs, cpp_srcs, cu_srcs = [], [], []
        for src in ext.sources:
            if src.endswith(".c"):
                c_srcs.append(src)
            elif src.endswith(".cpp"):
                cpp_srcs.append(src)
            elif src.endswith(".cu") and HAS_CUDA:
                cu_srcs.append(src)

        objs = []

        for src in cu_srcs:
            obj = src + ".o"
            cmd = ["nvcc", "-O3", "-arch=sm_80", "-Xcompiler", "-fPIC", "-DCUDA_AVAILABLE"]
            for d in include_dirs:
                cmd += ["-I", d]
            subprocess.check_call(cmd + ["-c", src, "-o", obj])
            objs.append(obj)

        py_includes = subprocess.check_output(["python3-config", "--includes"]).decode().split()

        base_cflags = ["-O3", "-fPIC", "-march=native", "-fopenmp"]
        if HAS_CUDA:
            base_cflags.append("-DCUDA_AVAILABLE")

        for src in c_srcs:
            obj = src + ".o"
            cmd = ["cc"] + base_cflags
            for d in include_dirs:
                cmd += ["-I", d]
            cmd += py_includes + ["-c", src, "-o", obj]
            subprocess.check_call(cmd)
            objs.append(obj)

        for src in cpp_srcs:
            obj = src + ".o"
            cmd = ["g++"] + base_cflags
            for d in include_dirs:
                cmd += ["-I", d]
            cmd += py_includes + ["-c", src, "-o", obj]
            subprocess.check_call(cmd)
            objs.append(obj)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        linker = "g++"

        if HAS_CUDA:
            link = ["nvcc", "-shared", "-o", ext_path] + objs
            link += ["-L/usr/local/cuda/lib64", "-lcudart", "-lgomp"]
        else:
            link = [linker, "-shared", "-o", ext_path] + objs + ["-lgomp", "-fPIC"]
        link += ["-I" + pybind11.get_include()] + py_includes
        subprocess.check_call(link)


c_sources = [
    "src/arena.c", "src/arena_cpu.c", "src/graph.c", "src/node.c", "src/op.c", "src/tensor.c",
    "src/kernels/cpu/abs.c", "src/kernels/cpu/add.c", "src/kernels/cpu/broadcast.c",
    "src/kernels/cpu/conv2d.c", "src/kernels/cpu/cos.c", "src/kernels/cpu/cpu_tensor_init.c",
    "src/kernels/cpu/cpu_utils.c", "src/kernels/cpu/div.c", "src/kernels/cpu/expand.c",
    "src/kernels/cpu/exp.c", "src/kernels/cpu/flatten.c", "src/kernels/cpu/iter.c", "src/kernels/cpu/leaky_relu.c",
    "src/kernels/cpu/log.c", "src/kernels/cpu/matmul.c", "src/kernels/cpu/pack.c", "src/kernels/cpu/max.c",
    "src/kernels/cpu/mean.c", "src/kernels/cpu/min.c", "src/kernels/cpu/mul.c",
    "src/kernels/cpu/neg.c", "src/kernels/cpu/sin.c", "src/kernels/cpu/squeeze.c",
    "src/kernels/cpu/sub.c", "src/kernels/cpu/sum.c", "src/kernels/cpu/tan.c",
    "src/kernels/cpu/transpose.c", "src/kernels/cpu/unsqueeze.c", "src/kernels/cpu/view.c",
    "src/optimizers/cpu/adam.c", "src/optimizers/cpu/adamw.c", "src/optimizers/cpu/sgd.c",
    "src/optimizers/cpu/zero_grad.c",
    "src/python/bindings.cpp",
]

cu_sources = [
    "src/arena_cuda.cu", "src/kernels/cuda/abs.cu", "src/kernels/cuda/add.cu",
    "src/kernels/cuda/conv2d.cu", "src/kernels/cuda/cos.cu", "src/kernels/cuda/cuda_pack.cu", "src/kernels/cuda/cuda_tensor_init.cu",
    "src/kernels/cuda/div.cu", "src/kernels/cuda/exp.cu", "src/kernels/cuda/leaky_relu.cu",
    "src/kernels/cuda/log.cu", "src/kernels/cuda/matmul.cu", "src/kernels/cuda/max.cu",
    "src/kernels/cuda/mean.cu", "src/kernels/cuda/min.cu", "src/kernels/cuda/mul.cu",
    "src/kernels/cuda/neg.cu", "src/kernels/cuda/sin.cu", "src/kernels/cuda/sub.cu",
    "src/kernels/cuda/sum.cu", "src/kernels/cuda/tan.cu", "src/optimizers/cuda/adam.cu",
    "src/optimizers/cuda/adamw.cu", "src/optimizers/cuda/sgd.cu", "src/optimizers/cuda/zero_grad.cu",
]

setup(
    name="plast",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "plast.plast_core",
            sources=c_sources + (cu_sources if HAS_CUDA else []),
        )
    ],
    cmdclass={"build_ext": custom_build_ext},
    install_requires=["numpy"],
)
