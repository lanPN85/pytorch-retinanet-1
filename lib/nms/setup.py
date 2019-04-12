from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='pth_nms',
      ext_modules=[CppExtension('pth_nms', [
                'src/nms.cpp', 'src/nms.h', 'src/nms_cuda.cpp', 'src/nms_cuda.h',
                'src/cuda/nms_kernel.cu', 'src/cuda/nms_kernel.h']
            )
        ],
      cmdclass={'build_ext': BuildExtension})