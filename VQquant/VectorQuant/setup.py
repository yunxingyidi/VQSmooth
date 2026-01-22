from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='VectorQuant',
    ext_modules=[
        CUDAExtension(
            name='VectorQuant',            # 这个名字必须和你 import 一致
            sources=['vector_quant_launcher.cpp', 'vector_dequant.cu'],   # 你的 kernel + launcher 文件
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
            'torch',
            'pybind11'
        ]
)
