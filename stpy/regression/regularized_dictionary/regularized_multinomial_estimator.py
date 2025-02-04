from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary

class RegularizedMultinomialEstimator(RegularizedDictionary):
    def fit(self, group_indices, y, sum_dim=None):
        if group_indices.dim() == 2 and sum_dim is not None:
            raise ValueError("Cannot use sum_dim with (N,K) indices")
        if sum_dim is None and group_indices.dim() != 2:
            raise ValueError("Must specify sum_dim for indices with more than 2 dimensions")
            
        x = self.phi[group_indices]
        if sum_dim is not None:
            x = x.sum(dim=sum_dim)
        
        # Final shape check
        if x.shape[:2] != y.shape:
            raise ValueError(f"Feature shape {x.shape} doesn't match label shape {y.shape}")
        
        data = (x, y)
        self.likelihood.load_data(data)
        self.calculate()

    def mean(self, xtest):
        if xtest.device != self.theta_ml().device:
            xtest = xtest.to(self.theta_ml().device)
        return super().mean(xtest)
    
