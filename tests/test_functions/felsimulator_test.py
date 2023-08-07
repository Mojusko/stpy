import torch
import numpy as np
from stpy.test_functions.swissfel_simulator import FelSimulator
from stpy.test_functions.benchmarks import SwissFEL
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.helper import interval_torch

if __name__ == "__main__":
    sigma = 0.1
    xtest = interval_torch(30, d= 2, L_infinity_ball=0.5)
    F = SwissFEL(d =2, dts = 'evaluations_bpm.hdf5')
    F.Simulator.GP.visualize_contour(xtest)


