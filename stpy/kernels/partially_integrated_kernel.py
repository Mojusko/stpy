from stpy.kernels import KernelFunction
from stpy.borel_set import BorelSet
from stpy.helpers.helper import cartesian
from stpy.regression.nystrom_fea import NystromFeatures
import torch
class IntegratedKernel(KernelFunction):

    def squared_exponential_kernel(self, data1, data2, **kwargs):
        """

        :param data1:
        :param data2:
        :param kwargs:
        :return:
        """
        m = len(data1)
        n = len(data2)

        q = 128
        K = torch.zeros((m,n)).double()
        for i,d1 in enumerate(data1):
            for j,d2 in enumerate(data2):

                if d1[2] is None and d2[2] is None:
                    K[i,j] = super().squared_exponential_kernel(d1[0],d2[0])

                elif d1[2] is not None and d2[2] is None:
                    for integral,sign in d1[0]:
                        weights, nodes = integral.return_legendre_discretization(q)
                        K[i, j] += torch.sum(weights * super().squared_exponential_kernel(nodes, d2[0]))*sign

                elif d1[2] is None and d2[2] is not None:
                    for integral,sign in d2[0]:
                        weights, nodes = integral.return_legendre_discretization(q)
                        K[i, j] += sign* torch.sum(weights * super().squared_exponential_kernel(d1[0], nodes))

                else:
                    for integral1,sign1 in d1[0]:
                        for integral2,sign2 in d2[0]:
                            weights, nodes = integral1.return_legendre_discretization(q)
                            weights2, nodes2 = integral2.return_legendre_discretization(q)

                            weights = torch.prod(torch.from_numpy(cartesian([weights, weights2])), dim=1)
                            nodes = torch.from_numpy(cartesian([nodes, nodes2]))

                            vals = super().squared_exponential_kernel(nodes[:, 0].view(-1, 1), nodes[:, 1].view(-1, 1))
                            K[i, j] += sign1*sign2*torch.sum(weights * vals)
        return K


if __name__ == "__main__":
    S = BorelSet(1, torch.tensor([[0,1]]).double())
    S2 = BorelSet(1, torch.tensor([[0,0.5]]).double())
    x = torch.tensor([[0.1]]).double()
    x2 = torch.tensor([[0.2]]).double()
    y =  torch.tensor([[0.2]]).double()
    data = [(x, y, None),(x2, y, None),([(S,1),(S2,-1)], y, 1),([(S2,1)], y, 1)]

    k = IntegratedKernel()
    print (k.kernel(data, data))


    embed = lambda q: k.kernel(q, data).T

    GP = NystromFeatures(m = 4, kernel_object =k )
    GP.fit_gp(data, None)