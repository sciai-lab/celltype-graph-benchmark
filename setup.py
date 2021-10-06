from setuptools import setup, find_packages

exec(open('pctg_benchmark/__version__.py').read())
setup(
    name='pctg_benchmark',
    version=__version__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    description='',
    author='',
    url='',
    author_email='',
)
