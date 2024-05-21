# Stochastic Process Library for Python (stpy)

<img src="stpy.png" alt="Repository Icon" width="200">

`stpy` is a python library and allows for efficient and easy implentation of fitting, analysis and uncertainty quantification of machine learning models. Particular emphasis is put on 
- Gaussian process (various kernels and kernel arirthmetic supported)
- Embedding approximation such as
  - Fourier features
  - Nystrom features
  - Bernstein polynomials 
  - Different polynomial features
  - shape constrained features
- Poisson Point Process & Cox Point Process
- Multiple Kernel Learning
- Dirichlet & Categorical Mixtures

We are inspired by [`scikit-learn`](), a influential python package, and we implement it with procedures `add_data` and `fit` workflow. This library is direct competitor to [`gpytorch`](https://gpytorch.ai/) where our library is easier, according to our opinion, easier to use and optimize hyperparameters.

## Installation
First clone the repository:

`git clone https://github.com/Mojusko/stpy`

Inside the project directory, run

`pip install -e .`

The `-e` option installs the package in "editable" mode, where pip links to your local copy of the repository, instead of copying the files to the your site-packages directory. That way, a simple `git pull` will update the package.
The project requires Python 3.6, and the dependencies should be installed with the package (let me know if it doesn't work for you).

## Tutorials
  - Gaussian process
     1. Fitting the model through evidence maximization [ref](tutorials/model_selection_marginalized_likelihood.py.ipynb)
     2. Comparison of sampling with Fourier Features [ref](tutorials/fourier-features.ipynb) 
  - Poisson point process 
    1. Fitting a poisson process 
  - Regularized Kernel Ridge Regression
    1. fitting & creating uncertainty sets
    2. Setting a basis 
## Requires
  - Classical: pytorch, cvxpy, numpy, scipy, sklearn, pymanopt, mosek, pandas
  
  - Special: 
        1. pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>
        
## Projects using `stpy`
1. [`sensepy`](https://github.com/Mojusko/sensepy) - poisson sensing library 
2. [`mutedpy`](https://github.com/Mojusko/mutedpy) - active learning and analysis of mutational data for protein engineering 
3. [`doexpy`](https://github.com/Mojusko/doexpy) - modern experiment design library

## Licence
Copyright (c) 2021 Mojmir Mutny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software to use the Software but not distribute further.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Contributions
Mojmir Mutny
