from setuptools import setup, find_packages

exec(open('ctg_benchmark/__version__.py').read())
setup(
    name='ctg_benchmark',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    description='PlantCellType Graph Benchmark',
    author='Lorenzo Cerrone',
    url='https://github.com/hci-unihd/celltype-graph-benchmark',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)
