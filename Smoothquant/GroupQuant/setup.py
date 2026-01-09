from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GroupQuant',
    ext_modules=[
        CUDAExtension(
            name='GroupQuant',            # 这个名字必须和你 import 一致
            sources=['group_quant.cu', 'group_quant_launcher.cpp', 'group_dequant.cu'],   # 你的 kernel + launcher 文件
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
            'torch',
            'pybind11'
        ]
)
