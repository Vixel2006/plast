import os
import subprocess
import sys
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# Function to find pybind11 include directory
def get_pybind11_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        raise RuntimeError("pybind11 must be installed to build the C++ extension.")

def has_cuda():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

class CMakeBuild(build_ext):
    def run(self):
        build_directory = os.path.abspath(self.build_temp)
        os.makedirs(build_directory, exist_ok=True)

        # --- Build SLEEF ---
        sleef_source_dir = os.path.abspath(os.path.join("vendor", "sleef"))
        sleef_build_dir = os.path.join(build_directory, "sleef_build")
        os.makedirs(sleef_build_dir, exist_ok=True)

        print("Configuring and building SLEEF...")
        sleef_cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSLEEF_STATIC_LIBS=ON",
            "-DSLEEF_TEST=OFF",
            "-DSLEEF_SMP=OFF",
            "-DSLEEF_OPENMP=OFF",
        ]
        subprocess.check_call(["cmake", sleef_source_dir, *sleef_cmake_args], cwd=sleef_build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=sleef_build_dir)
        print("SLEEF built successfully.")

        # Locate static library
        sleef_static_lib_name = "libsleef.a" if sys.platform != "win32" else "sleef.lib"
        potential_paths = [
            os.path.join(sleef_build_dir, sleef_static_lib_name),
            os.path.join(sleef_build_dir, "lib", sleef_static_lib_name),
            os.path.join(sleef_build_dir, "Release", sleef_static_lib_name),
            os.path.join(sleef_build_dir, "src", "lib", sleef_static_lib_name),
        ]
        sleef_library_path = next((p for p in potential_paths if os.path.exists(p)), None)
        if not sleef_library_path:
            raise FileNotFoundError(f"SLEEF static lib not found in expected locations: {potential_paths}")

        sleef_include_dir = os.path.join(sleef_source_dir, "include")

        # --- Build Plast C++ Core and Pybind11 Module ---
        print("Configuring and building Plast C++ Core and Pybind11 Module...")
        plast_cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}",
            f"-DPYBIND11_INCLUDE_DIR={get_pybind11_include()}", # Pass pybind11 include path
            f"-DSLEEF_INCLUDE_DIR={sleef_include_dir}",
            f"-DSLEEF_LIBRARY={sleef_library_path}",
        ]

        if has_cuda():
            print("CUDA detected. Building with CUDA support.")
            plast_cmake_args.append("-DPLAST_BUILD_CUDA=ON")
        else:
            print("CUDA not detected. Building without CUDA support.")

        if sys.platform == "win32":
            axon_cmake_args.append("-GVisual Studio 17 2022")

        subprocess.check_call(["cmake", os.path.abspath("."), *plast_cmake_args], cwd=build_directory)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_directory)
        print("Axon C++ Core and Pybind11 Module built successfully.")

        # --- Copy built libraries into package ---
        # Copy the pybind11 module
        pybind_module_name = "_plast_cpp_core"
        if sys.platform == "linux":
            pybind_lib_pattern = f"{pybind_module_name}*.so"
        elif sys.platform == "darwin":
            pybind_lib_pattern = f"{pybind_module_name}*.dylib"
        elif sys.platform == "win32":
            pybind_lib_pattern = f"{pybind_module_name}*.pyd"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # Find the built pybind11 module
        built_pybind_module = None
        for f in os.listdir(build_directory):
            if f.startswith(pybind_module_name) and (f.endswith(".so") or f.endswith(".dylib") or f.endswith(".pyd")):
                built_pybind_module = os.path.join(build_directory, f)
                break

        if not built_pybind_module:
            raise FileNotFoundError(f"Built pybind11 module '{pybind_lib_pattern}' not found in {build_directory}")

        package_dir = os.path.join(os.path.abspath("."), "plast", "cpp_bindings")
        os.makedirs(package_dir, exist_ok=True)
        shutil.copyfile(built_pybind_module, os.path.join(package_dir, os.path.basename(built_pybind_module)))
        print(f"Copied {os.path.basename(built_pybind_module)} into plast/cpp_bindings package.")


# Custom install command to ensure build_ext runs before install
class CustomInstall(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)

setup(
    name="plast-dl",
    version="0.1.0",
    packages=find_packages(),
    cmdclass={"build_ext": CMakeBuild, "install": CustomInstall},
    setup_requires=["pybind11>=2.10"], # Ensure pybind11 is available for setup
    install_requires=["pybind11>=2.10"], # Ensure pybind11 is installed for runtime
    # package_data={"plast": get_platform_package_data()}, # This might be redundant if pybind11 module is the main output
    zip_safe=False,
)