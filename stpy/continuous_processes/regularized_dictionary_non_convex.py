import torch
from typing import Union
import cvxpy as cp
import numpy as np
import mosek
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import Embedding
from stpy.probability.likelihood import Likelihood

class RegularizedDictionaryNonConvex(RegularizedDictionary):

    def __init__(self, embedding: Embedding, likelihood: Likelihood, **kwargs):
        super().__init__(embedding, likelihood, **kwargs)

    def calculate(self):
        if not self.regularizer.is_convex():
            self.calculate_non_convex_lq()
        else:
            self.calculate_non_convex_set()

    def calculate_non_convex_lq(self, repeats=100):
        if self.regularizer.groups is None:
            eta = np.ones(self.m) * 1
        else:
            eta = np.ones(len(self.regularizer.groups)) * 1

        for _ in range(repeats):
            theta = cp.Variable((self.m, 1))
            likelihood = self.likelihood.get_objective_cvxpy()
            regularizer = self.regularizer.get_regularizer_cvxpy(eta)
            objective = likelihood(theta) + regularizer(theta)
            constraints = []
            if self.constraints is not None and self.use_constraint:
                set = self.constraints.get_constraint_cvxpy(theta)
                constraints += set

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                                                      mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
                                                      mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
                                                      mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
            if self.regularizer.groups is None:
                eta = np.abs(theta.value) ** (2 - self.regularizer.q)
            else:
                eta = np.array(
                    [np.linalg.norm(theta.value[group]) ** (2 - self.regularizer.q) for group in self.regularizer.groups])
            eta = eta + 1e-8

        # print (theta.value)
        self.theta_fit = torch.from_numpy(theta.value)
        self.fitted = True
        return theta.value


    def calculate_non_convex_set(self):
        if self.constraints is not None and self.use_constraint:
            theta = cp.Variable((self.m, 1))
            likelihood = self.likelihood.get_objective_cvxpy()
            if self.regularizer is not None:
                objective = likelihood(theta) + self.regularizer.get_regularizer_cvxpy()(theta)
            else:
                objective = likelihood(theta)

            values = []
            arg_values = []
            for con in self.constraints.get_list_cvxpy_constraints(theta):
                constraints = []
                constraints += con

                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                                                          mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
                                                          mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
                                                          mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
                values.append(prob.value)
                arg_values.append(theta.value)
            self.theta_fit = torch.from_numpy(arg_values[np.argmin(values)])
            self.fitted = True


    def objective_on_non_convex_confidence_set(self, theta, objective, type=None):
        set = self.likelihood.get_confidence_set_cvxpy(theta, type=type,
                                                       information=[self.theta_fit,
                                                                    (1e-4) * torch.diag(
                                                                        torch.ones(size=theta.shape).view(-1)).double()])
        objective = objective
        set_of_constraints = self.constraints.get_list_cvxpy_constraints(theta)
        values = []
        args = []
        for con in set_of_constraints:
            prob = cp.Problem(cp.Minimize(objective), con + set)
            prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=
            {mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
             mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
             mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
             mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
            values.append(prob.value)
            args.append(theta.value)
        index = np.argmin(values)
        return np.min(values), torch.from_numpy(args[index])


    def objective_on_non_convex_confidence_set_bisection(self, theta, objective, type=None):
        def optimize_for_lam(lam, self, objective, theta):

            if self.regularizer.groups is None:
                eta = np.ones(self.m) * 1
            else:
                eta = np.ones(len(self.regularizer.groups)) * 1
            repeats = 3

            for _ in range(repeats):
                # theta = cp.Variable((self.m, 1))
                set = self.likelihood.get_confidence_set_cvxpy(theta, type=type,
                                                               information=[self.theta_fit,
                                                                            self.regularizer.hessian(self.theta_fit)])
                regularizer = self.regularizer.get_regularizer_cvxpy(eta)
                objective = objective + lam * (regularizer(theta) - 1)
                constraints = set

                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                                                          mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
                                                          mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
                                                          mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
                if self.regularizer.groups is None:
                    eta = np.abs(theta.value) ** (2 - self.regularizer.q)
                else:
                    eta = np.array([np.linalg.norm(theta.value[group]) ** (2 - self.regularizer.q) for group in
                                    self.regularizer.groups])
                eta = eta + 1e-8
            value = prob.value
            return value

        optimize_for_lam_small = lambda a: optimize_for_lam(a, self, objective, theta)
        # lam_final = bisection(optimize_for_lam_small,1e-5,10,100)
        return optimize_for_lam_small(1.0), None


    def lcb(self, xtest: torch.Tensor, arg=False, sign=1.):
        args = []
        n = xtest.size()[0]
        values = torch.zeros(size=(n, 1)).double()
        Phi = self.embed(xtest)

        for j in range(n):

            theta = cp.Variable((self.m, 1))
            objective = sign * Phi[j, :] @ theta

            if (self.constraints is not None and not self.constraints.is_convex()):
                # non-convex set
                value, theta_lcb = self.objective_on_non_convex_confidence_set(theta, objective, type=self.inference_type)
            elif not self.regularizer.is_convex():
                # non-convex regularizer
                value, theta_lcb = self.objective_on_non_convex_confidence_set_bisection(theta, objective,
                                                                                         type=self.inference_type)
            else:
                # convex regularizer
                value, theta_lcb = self.objective_on_confidence_set(theta, objective, inference_type=self.inference_type)

            values[j] = sign * value
            if arg:
                args.append(theta_lcb)

        if args:
            return values, args
        else:
            return values