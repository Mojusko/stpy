from abc import ABC, abstractmethod
from typing import Callable
from stpy.helpers.helper import cartesian
import cvxpy as cp
import numpy as np
import torch
import copy
import cdd


class Constraints(ABC):
    def __init__(self):
        self.convex = True
        pass

    def is_convex(self):
        return self.convex

    @abstractmethod
    def get_constraint_cvxpy(self, theta):
        pass


class CustomConstraint(Constraints):

    def __init__(self, custom_function: Callable, custom_function_cvxpy: Callable):
        super().__init__()
        self.fn = custom_function
        self.fn_cvxpy = custom_function_cvxpy

    def get_constraint_cvxpy(self, theta):
        return self.fn_cvxpy(theta)


class LinearEqualityConstraint(Constraints):

    def __init__(self):
        pass


class LinearInequalityConstraint(Constraints):

    def __init__(self):
        pass


class AbsoluteValueConstraint(Constraints):

    def __init__(self, c=None):
        if c is None:
            self.c = 1.
        else:
            self.c = c

    def get_constraint_cvxpy(self, theta):
        set = [cp.norm1(theta) <= self.c]
        return set


class QuadraticInequalityConstraint(Constraints):
    """
    xQx - b@x <= c
    """

    def __init__(self, Q, b=None, c=None):
        self.Q = Q
        if c is None:
            self.c = 1.
        else:
            self.c = c
        if b is None:
            self.b = torch.zeros(size=(Q.size()[0], 1))
        else:
            self.b = b

    def get_constraint_cvxpy(self, theta):
        set = [cp.quad_form(theta, self.Q) - self.b.T @ theta <= self.c]
        return set


class NonConvexGroupNormConstraint(Constraints):
    def __init__(self, q, c, d, groups):
        super().__init__()
        self.q = q
        self.c = c
        self.d = d
        self.groups = groups
        self.convex = False


    def get_list_cvxpy_constraints(self, theta):
        w = self.q / (1 - self.q)
        set_of_constraints = []
        d = len(self.groups)
        for i in range(d):

            # l1 constraint
            constraints = []
            weights = np.ones(d) * w
            weights[i] = 1.
            group = self.groups[i]
            constraints.append(cp.norm(theta[group]).T * weights[i] <= self.c)
            # l_infinity constraint
            for j in range(d):
                if i != j:
                    group = self.groups[j]
                    constraints.append(cp.norm(theta[group]) <= self.q * self.c)
            group = self.groups[i]
            constraints.append(cp.norm(theta[group]) <= self.c)
            set_of_constraints.append(constraints)
        return set_of_constraints

    def get_constraint_cvxpy(self, theta):
        ## Does not work for non-convex constraints
        return None

class NonConvexNormConstraint(Constraints):

    def __init__(self, q, c, d):
        super().__init__()
        self.q = q
        self.c = c
        self.d = d
        self.convex = False
        # self.construct(q,d)

    def construct(self, q, d):
        self.vertex_description = []
        self.polyhedra_vertex_description = []
        square = cartesian([[-q, q] for _ in range(d)])
        for i in range(2 * d):
            polytope = copy.copy(square)
            zero = np.zeros(d).reshape(1, -1)
            appex = copy.copy(zero)
            appex[0, i // 2] = (float(i % 2) - 0.5) * 2.
            polytope = np.concatenate((appex, polytope))
            self.vertex_description.append(polytope)
            self.polyhedra_vertex_description.append(polytope)
        # print (self.vertex_description)

        self.modify_to_inequality()

    def modify_to_inequality(self):
        self.polyhedra_inequality_description = []
        for polyhedra in self.polyhedra_vertex_description:
            t = np.ones(polyhedra.shape[0]).reshape(-1, 1)
            vertex_description = np.concatenate([t, polyhedra], axis=1)
            mat = cdd.Matrix(vertex_description.tolist())
            mat.rep_type = cdd.RepType.GENERATOR
            mat.canonicalize()
            poly = cdd.Polyhedron(mat)
            ext = poly.get_inequalities()
            inequality_description = np.array(list(ext))
            b = inequality_description[:, 0].reshape(-1, 1)
            A = inequality_description[:, 1:]
            self.polyhedra_inequality_description.append([A, b])

    def get_constraint_cvxpy(self, theta):
        ## Does not work for non-convex constraints
        return None

    def get_list_cvxpy_constraints(self, theta):
        w = self.q / (1 - self.q)
        set_of_constraints = []
        for i in range(self.d):

            # l1 constraint
            constraints = []
            weights = np.ones(self.d) * w
            weights[i] = 1.
            constraints.append(cp.abs(theta).T @ weights <= self.c)
            # l_infinity constraint
            for j in range(self.d):
                if i != j:
                    constraints.append(cp.abs(theta[j]) <= self.q * self.c)
            constraints.append(cp.abs(theta[i]) <= self.c)
            set_of_constraints.append(constraints)
        return set_of_constraints

    def eval(self, theta):
        out = []
        for [A, b] in self.polyhedra_inequality_description:
            out.append((A @ theta.T <= b).all())
        return max(out)


if __name__ == "__main__":
    c = NonConvexNormConstraint(0.5, 1, 2)
    point = np.zeros(2).reshape(1, -1)
