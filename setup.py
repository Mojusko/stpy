from setuptools import setup

packages = [
    'numpy',
    'scipy',
    'matplotlib',
    'sklearn',
    'tensorflow',
    'cvxpy',
    'torch',
    'pymanopt',
    'pandas',
    'mosek',
]
#
setup(name='stpy',
      version='0.0.2',
      description='Stochastic Process Library for Python',
      url='',
      author='Mojmir Mutny',
      author_email='mojmir.mutny@inf.ethz.ch',
      license='custom ',
      packages=['stpy'],
	    zip_safe=False,
      install_requires=packages)
