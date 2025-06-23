from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="functions_SBM_casimodo",
        sources=["functions_SBM_casimodo.pyx"],
        extra_compile_args=[
            "-std=c99",
            "-O3",              # Maximize speed optimization
            "-ffast-math",      # Allow unsafe FP math optimizations
            "-march=native",    # Optimize for your CPU architecture
            "-flto",            # Link-time optimization
            "-funroll-loops",   # Unroll loops for speed
            "-fomit-frame-pointer", # Omit frame pointer (improves speed)
            "-fstrict-aliasing" # Allow strict aliasing optimizations
        ],
        extra_link_args=[
            "-flto",
        ],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]  # Use modern NumPy API
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,    # Disable bounds checks
            "wraparound": False,     # Disable negative index wraparound
            "cdivision": True,       # Enable C division semantics
            "nonecheck": False,      # Disable None checks
            "initializedcheck": False,  # Disable uninitialized variable check
            "language_level": 3,     # Use Python 3 syntax
            "infer_types": True      # Infer C types for local variables (if supported)
        },
        annotate=False,  # Set to True if you want an HTML report of Cython optimizations
        language_level=3,
        # no profiling, no linetracing => faster
    ),
    zip_safe=False,
)