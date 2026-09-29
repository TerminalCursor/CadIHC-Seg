from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path

extensions = []
for f in Path('src').rglob('*.pyx'):
    mod_name = str(f.relative_to('src')).replace('/', '.').replace('\\', '.')[:-4]
    extensions.append(Extension(mod_name, [str(f)], include_dirs=[numpy.get_include()]))

setup(
    name="histseg",
    ext_modules=cythonize(extensions, include_path=["src"], compiler_directives={'language_level': "3"}),
    packages=["histseg"],
    package_dir={"": "src"},
)
