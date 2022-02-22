from setuptools import setup, find_packages

exec(open('ctg_benchmark/__version__.py').read())
setup(
    name='ctg_benchmark',
    version=__version__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    description='PlantCellType Graph Benchmark',
    author='Anonymous',
    url='TODO',
    author_email='TODO',
)
