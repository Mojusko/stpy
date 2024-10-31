import copy

import torch
import scipy
import numpy as np
from typing import Union
import mosek
import cvxpy as cp
from stpy.regression.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import Embedding
from stpy.probability.likelihood import Likelihood
from stpy.optim.custom_optimizers import bisection
from stpy.regularization.sdp_constraint import SDPConstraint
from stpy.regression.nystrom_fea import NystromFeatures


class RegularizedDictionaryPSD(RegularizedDictionary):

    def __init__(self,
                 embedding: Embedding,
                 likelihood: Likelihood,
                 sdp_constraint: Union[SDPConstraint, None] = None,
                 pattern='full',
                 norm_regularization = False,
                 min_bound = 0.,
                 B = 10,
                 **kwargs):

        super().__init__(embedding, likelihood, **kwargs)

        self.sdp_regularizer = sdp_constraint
        self.pattern = pattern
        self.min_bound = min_bound
        self.tightness = 1e-3
        self.B = B
        self.norm_regularization = norm_regularization

        self.mosek_opts = {mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
                        mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
                        mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
                        mosek.dparam.intpnt_co_tol_rel_gap: 1e-6}

    def bias_enumerate_2(self, x, add_data = True):
        phi = self.embed(x)
        V = self.likelihood.information_matrix()
        vals = []
        vals_c = []
        v = self.theta_fit.T @ torch.linalg.solve(self.A_fit,self.theta_fit)

        for i in range(self.m):
            a = cp.Variable(self.m)
            t = cp.Variable(1)
            theta_new = cp.Variable(self.m)
            lams = np.diag(V)
            objective = phi**2 @ a
            constraints = [cp.sum(a) <= a[i] * self.sdp_regularizer.matrix_bound] + \
                          [a[i] >= a[j] for j in range(self.d)] + \
                          [a >= 0] + \
                          [a[i] <= self.sdp_regularizer.lambda_max_constraint*t] + \
                          [t >= 0 ] + \
                          [lams@a + self.m*t == 1]

            if add_data:
                constraints += [a >= t*(self.theta_fit.view(-1)**2)/self.m]

                prob = cp.Problem(cp.Maximize(objective), constraints)
                prob.solve(solver=cp.MOSEK,
                           verbose=False)

                a_old = a.value / t.value
                val = phi ** 2 @ a_old / (lams @ a_old + self.m)
                val_c = self.theta_fit.T @ np.diag(a_old) @ self.theta_fit, v

            else:
                from scipy.optimize import linprog
            vals.append(val)
            vals_c.append(val_c)
        #print (vals_c[np.argmax(vals)], vals[np.argmax(vals)], v)
        return np.max(vals) * self.B

    def bias_enumerate(self,x):
        phi = self.embed(x)
        V = self.likelihood.information_matrix()
        A_bar = self.A_fit + torch.eye(self.m) * 0.
        if not self.fitted_bias:
            vals = []

            for i in range(self.m):
                a = cp.Variable(self.m)
                A = cp.diag(a)
                objective = cp.trace(A @ (A_bar + torch.eye(self.m)))

                constraints = [a >= self.min_bound] + [cp.trace(A) <= a[i] * self.sdp_regularizer.matrix_bound]+\
                 [cp.max(a) <= a[i], cp.max(a) <= self.sdp_regularizer.lambda_max_constraint]

                prob = cp.Problem(cp.Maximize(objective), constraints)

                prob.solve(solver=cp.MOSEK,
                           mosek_params=self.mosek_opts,
                           verbose=False)

                vals.append(prob.value)

            self.fitted_bias = True
            self.fit_bias_value = np.max(vals)



        L = torch.from_numpy(scipy.linalg.sqrtm(A_bar.detach().numpy()).real)
        I = torch.eye(self.m)
        err_x = phi @ L@ torch.linalg.inv(L@V@L.T + I)@L @ phi.T
        print (err_x, self.fit_bias_value)
        err = 2 * err_x * (self.fit_bias_value + 1)

        return err

    def bias(self, x, type='map'):

        phi = self.embed(x)
        all_phi = self.phi
        V = all_phi.T @ all_phi

        if self.sdp_regularizer.get_type() == "stable-rank":

            if self.pattern == "diagonal":
                return self.bias_enumerate(x)

            if type == 'map':
                A = torch.linalg.pinv(self.A_fit)
                L = torch.from_numpy(scipy.linalg.sqrtm(A.detach().numpy()).real)
                err_x = phi @ L @ torch.linalg.pinv(all_phi.T @ all_phi + A) @ L @ phi.T

                if not self.fitted_bias:
                    def calc(s_value):
                        if self.pattern == "diagonal":
                            a = cp.Variable(self.m)
                            A = cp.diag(a)
                        else:
                            A = cp.Variable((self.m, self.m))
                        l = cp.Variable(1)
                        objective = cp.trace(A @ (self.A_fit + torch.eye(self.m)))
                        constraints = [A >> self.min_bound] + [cp.trace(A) <= self.sdp_regularizer.matrix_bound * l] + [
                            cp.lambda_max(A) <= l] + [l <= s_value]
                        prob = cp.Problem(cp.Maximize(objective), constraints)
                        prob.solve(solver=cp.MOSEK, mosek_params=self.mosek_opts, verbose=False)
                        return prob.value, torch.from_numpy(l.value), torch.max(
                            torch.linalg.eigvalsh(torch.from_numpy(A.value)))

                    def fun(s):
                        _, a, b = calc(s)
                        print (a,b)
                        return -(-self.tightness + a - b)

                    #res = bisection(fun, 10e-3, 10e3, 15)
                    #print("Bisection", res)

                    bias = calc(s_value=self.l_fit)[0]

                    self.fit_bias_value = bias
                    self.fitted_bias = True
                err = self.fit_bias_value * err_x
            else:

                A = cp.diag(cp.Variable(self.m))
                objective = cp.trace(phi.T @ phi @ A)

                # not sure if the constraint fits
                constraints = [A >> self.min_bound]
                constraints += [A << torch.linalg.pinv(V + 1 * torch.eye(self.m))]
                # constraints += [cp.trace(A) <= cp.trace(np.linalg.pinv(torch.eye(self.m) / self.m - V))]

                prob = cp.Problem(cp.Maximize(objective), constraints)
                prob.solve(solver=cp.MOSEK, mosek_params=self.mosek_opts)
                err = prob.value
        else:
            if type == 'map':
                A = torch.linalg.pinv(self.A_fit)
                L = torch.from_numpy(scipy.linalg.sqrtm(A.detach().numpy()).real)
                err_x = torch.diag(phi @ L @ torch.linalg.pinv(all_phi.T @ all_phi + A) @ L @ phi.T)

                if not self.fitted_bias:
                    A = cp.Variable((self.m, self.m))
                    objective = cp.trace(A @ (self.A_fit + torch.eye(self.m)))
                    constraints = [A >> self.min_bound, cp.trace(A) <= self.sdp_regularizer.trace_constraint]
                    prob = cp.Problem(cp.Maximize(objective), constraints)
                    prob.solve(solver=cp.MOSEK, mosek_params=self.mosek_opts, verbose=False)
                    self.fit_bias_value = prob.value
                    self.fitted_bias = True

                err = err_x # self.fit_bias_value * err_x


            elif type == 'worst-case':
                A = cp.Variable((self.m, self.m))

                objective = cp.trace(phi.T @ phi @ A)
                # # not sure if the constraint fits
                constraints = [A >> self.min_bound]
                constraints += [A << torch.linalg.pinv(V)]
                constraints += [cp.trace(A) <= self.sdp_regularizer.trace_constraint * cp.trace(
                    np.linalg.pinv(torch.eye(self.m) / self.m - V))]

                prob = cp.Problem(cp.Maximize(objective), constraints)
                prob.solve(solver=cp.MOSEK, mosek_params=self.mosek_opts)
                err = prob.value
        err = err * self.bound ** 2
        return err

    def calculate(self):

        if self.fitted:
            if self.verbose:
                print("Skip fitting.")
            return

        if self.pattern == "diagonal" and self.sdp_regularizer.get_type() == "stable-rank":
            self.calculate_enumerate()

        elif self.sdp_regularizer is not None:
            if self.sdp_regularizer.get_type() == "stable-rank":
                # first output is l_value, eigenvalue
                def fun(s):
                    _, a, b = self.calculate_simple(s_value=s)
                    print (a,b)
                    return -(-self.tightness + a - b)

                res = bisection(fun, 10e-5, 10e5, 30)
                self.fitted = True
                self.calculate_simple(s_value=res)

            else:
                self.calculate_simple(s_value=None)
                self.fitted = True
        else:
            self.calculate_simple(s_value=None)
        self.fitted = True
    def calculate_enumerate(self):
        assert (self.pattern == "diagonal")
        print ("Fitting using enummeration:")
        print ("---------------------------")
        theta = cp.Variable((self.m, 1))
        l = cp.Variable((1, 1))

        vals = []
        for i in range(self.m):
            a = cp.Variable((self.m, 1))
            A = cp.diag(a)

            likelihood = self.likelihood.get_objective_cvxpy()

            if self.norm_regularization:
                objective = likelihood(theta) + cp.matrix_frac(theta, A)
                constraints = []
            else:
                constraints = [cp.matrix_frac(theta, A) <= 1]
                objective = likelihood(theta)

            if self.regularizer is not None:
                regularizer = self.regularizer.get_regularizer_cvxpy()
                objective += regularizer(theta)


            q = self.sdp_regularizer.matrix_bound
            if self.verbose:
                print (i, "q:",q)

            constraints+= [ cp.max(a) <= a[i], a >= self.min_bound,
                            cp.trace(A) <= a[i]*q]\
                          + self.sdp_regularizer.get_constraint_cvxpy(A,l,None)

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.MOSEK)

            vals.append(prob.value)

        j = np.argmin(vals)
        if self.norm_regularization:
            constraints = [cp.max(a) <= a[j],
                           a >= self.min_bound, cp.trace(A) <= a[j]*q]\
                          + self.sdp_regularizer.get_constraint_cvxpy(A,l,None)
        else:
            constraints = [cp.matrix_frac(theta, A) <= 1, cp.max(a) <= a[j],
                           a >= self.min_bound, cp.trace(A) <= a[j] * q] \
                          + self.sdp_regularizer.get_constraint_cvxpy(A, l, None)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK)
        self.A_fit = torch.from_numpy(A.value)
        self.theta_fit = torch.from_numpy(theta.value)
        self.l_fit = None

        self.fitted = True
        print ("===========================")

        return prob.value, None, torch.max(torch.linalg.eigvalsh(torch.from_numpy(A.value)))

    def calculate_simple(self, s_value=None):
        theta = cp.Variable((self.m, 1))
        l = cp.Variable((1, 1))

        if self.pattern == "full":
            A = cp.Variable((self.m, self.m))

        elif self.pattern == "diagonal":
            a = cp.Variable(self.m)
            A = cp.diag(a)

        elif self.pattern == "block-diagonal":
            pass

        else:
            raise NotImplementedError("This pattern for PSD estimator is not implemented.")

        likelihood = self.likelihood.get_objective_cvxpy()

        if self.norm_regularization:
            objective = likelihood(theta) + cp.matrix_frac(theta, A)
            constraints = [A >> self.min_bound]
        else:
            objective = likelihood(theta)
            constraints = [cp.matrix_frac(theta, A) <= 1, A >> 0]

        objective = objective + self.sdp_regularizer.get_regularization_cvxpy(A,l,s_value)

        if self.regularizer is not None:
            regularizer = self.regularizer.get_regularizer_cvxpy()
            objective += regularizer(theta)

        if self.constraints is not None and self.use_constraint:
            set = self.constraints.get_constraint_cvxpy(theta)
            constraints += set

        constraints += self.sdp_regularizer.get_constraint_cvxpy(A, l, s_value)
        prob = cp.Problem(cp.Minimize(objective), constraints)


        prob.solve(solver=cp.MOSEK, mosek_params=self.mosek_opts, verbose=True)
        self.A_fit = torch.from_numpy(A.value)
        self.theta_fit = torch.from_numpy(theta.value)
        if l.value is None:
            self.l_fit = None
        else:
            self.l_fit = torch.from_numpy(l.value)

        return prob.value, l.value, float(torch.max(torch.linalg.eigvalsh(torch.from_numpy(A.value))))

    def lcb_enumerate(self, x: torch.Tensor, sign: float = 1., delta = None):

        if delta is None:
            delta = 0.01
        lcb = torch.zeros(size = (x.size()[0],1)).double()
        z = self.likelihood.get_objective_torch()(self.theta_fit).numpy()
        print ("Z",z )

        N = x.size()[0]
        for i in range(N):
            #print (torch.cdist(x[i,:].reshape(1,-1),self.x))
            if torch.min(torch.cdist(x[i,:].reshape(1,-1),self.x))>1e-2:
                m = self.m + 1
                embedding = NystromFeatures(kernel_object=self.embedding.kernel_object, m=m, approx = 'svd')
                embedding.fit_gp(torch.vstack((self.x,x[i,:].reshape(1,-1))), None, eps = 1e-5)

            else:
                m = self.m
                embedding = NystromFeatures(kernel_object=self.embedding.kernel_object, m=m, approx = 'svd')
                embedding.fit_gp(self.x, None, eps = 1e-5)

            likelihood = copy.deepcopy(self.likelihood)
            likelihood.load_data((embedding.embed(self.x), self.y))

            phi = embedding.embed(x[i, :].reshape(1, -1)).detach().numpy()
            theta = cp.Variable((m, 1))
            a = cp.Variable(m)
            objective = sign * theta.T @ phi.T
            vals = []

            for j in range(self.m):
                constraints = [cp.matrix_frac(theta, cp.diag(a)) <= 1, a >= self.min_bound] \
                              + [cp.sum(a) <= self.sdp_regularizer.matrix_bound * a[j]] + \
                              [a[j]>=cp.max(a),
                               cp.max(a) <= self.sdp_regularizer.lambda_max_constraint]

                constraints += [likelihood.get_objective_cvxpy()(theta) <= z + np.log(1./delta)]

                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.MOSEK,
                           verbose=False,
                           warm_start=False,
                           mosek_params=self.mosek_opts)

                vals.append(prob.value)

            lcb[i] = sign * np.min(vals)
            print (i, lcb[i])
        self.fit()
        return lcb
    def lcb(self, x: torch.Tensor, sign: float = 1., delta = None):
        """
        Lower confidence bound for the PSD estimator.
        :param x:
        :param sign:
        :return:
        """
        theta = cp.Variable((self.m, 1))
        l = cp.Variable((1, 1))

        if self.pattern == "full":
            A = cp.Variable((self.m, self.m))

        elif self.pattern == "diagonal":
            a = cp.Variable(self.m)
            A = cp.diag(a)
            if self.sdp_regularizer.type == "stable-rank":
                return self.lcb_enumerate(x, sign = sign, delta = delta)

        elif self.pattern == "block-diagonal":
            pass

        lcb = torch.zeros(size = (x.size()[0],1)).double()
        Phi = self.embed(x)

        if delta is None:
            delta = 0.1


        phi = cp.Parameter((1,self.m))
        for i in range(x.size()[0]):
            print ("i", i )
            phi.value = Phi[i,:].reshape(1,-1).numpy()
            objective = sign * theta.T @ phi.T
            constraints = [cp.matrix_frac(theta, A) <= 1, A >> 0]  + self.sdp_regularizer.get_constraint_cvxpy(A,l,None)

            constraints += [self.likelihood.get_objective_cvxpy()(theta)<=
                            self.likelihood.get_objective_torch()(self.theta_fit)+np.log(1./delta)]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.MOSEK, verbose=False, warm_start=True, mosek_params=self.mosek_opts)
            lcb[i] = sign * prob.value
        return lcb

