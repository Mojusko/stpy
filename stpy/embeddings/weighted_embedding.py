import numpy as np

from stpy.embeddings.embedding import Embedding


class WeightedEmbedding(Embedding):

    def __init__(self,
                 embedding: Embedding,
                 weights = None,
                 weight_function = None
                 ):
        self.base_embedding = embedding
        self.m = self.base_embedding.get_m()
        self.weights = weights
        self.weight_function = weight_function

    def weight(self):
        if self.weights is not None:
            pass
        elif self.weight_function is not None:
            pass

    def embed(self, xtest):
        self.weight()
        Phi = self.base_embedding.embed(xtest)

        if self.weights is not None:
            return Phi @ np.diag(self.weights)
        else:
            return Phi @ np.diag(self.weight_function(self.base_embedding))





