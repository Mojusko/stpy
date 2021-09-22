# Stochastic Process Library for Python (stpy)

## Installation
First clone the repository:

`git clone git@gitlab.inf.ethz.ch:mmutny/stpy.git`

Inside the project directory, run

`pip install -e .`

The `-e` option installs the package in "editable" mode, where pip links to your local copy of the repository, instead of copying the files to the your site-packages directory. That way, a simple `git pull` will update the package.
The project requires Python 3.6, and the dependencies should be installed with the package (let me know if it doesn't work for you).

## Tutorials
TBD

## Requires
  - Classical: pytorch, cvxpy, numpy, scipy, sklearn, pymanopt, mosek, pandas
  - Special: stpy (see: <>), pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>, hessian <https://github.com/mariogeiger/hessian>

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
