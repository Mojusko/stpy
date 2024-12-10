import torch
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.regularization.regularizer import L2Regularizer
from stpy.test_functions.benchmarks import Simple1DFunction
from stpy.helpers.helper import interval_torch
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator

from stpy.probability.multinomial_likelihood import MultinomialLikelihood

if __name__ == "__main__":
    n = 4024  # test points
    N = 2024    # base points to sample pairs from
    M = 4024    # number of pairs to sample
    K = 2     # pair size
    
    x = interval_torch(N, d=1)
    xtest = interval_torch(n, d=1)
    
    benchmark_f = Simple1DFunction(d=1)
    kappa = torch.max(torch.abs(benchmark_f.eval_noiseless(xtest)))
    
    # Set up embedding and estimator
    bound = 10
    lam = 1./bound
    embedding = HermiteEmbedding(m=128, gamma=0.1, d=1, kappa=kappa)
    likelihood = MultinomialLikelihood()
    regularizer = L2Regularizer(lam=lam)
    estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)
    
    # Generate M random pairs
    pair_indices = torch.randint(N, size=(M, K))
    # Get flattened pairs and evaluate
    flat_pairs = x[pair_indices].reshape(-1, 1)  # Reshape to (M*K, 1)
    f_values = benchmark_f.eval_noiseless(flat_pairs).reshape(M, K)  # Back to (M, K)

    y = torch.zeros((M, K))
    # Convert f_values to probabilities using softmax
    probs = torch.softmax(f_values, dim=1)
    # Sample from multinomial distribution for each row
    samples = torch.multinomial(probs, num_samples=1).squeeze()
    # Create one-hot encoded y
    y = torch.zeros((M, K))
    y[torch.arange(M), samples] = 1
    # New loading flow
    estimator.load_data((x, torch.zeros(N)))  # Load with dummy labels
    
    # Fit with indices and true labels
    estimator.fit(pair_indices, y)
   
    # Calculate preference error
    pair_values = estimator.mean(x[pair_indices].reshape(-1, 1)).reshape(M, K)
    predicted_preferences = torch.zeros((M, K))
    predicted_preferences[torch.arange(M), torch.argmax(pair_values, dim=1)] = 1
    preference_accuracy = (torch.sum(predicted_preferences == y).item() / (M * K)) * 100
    print(f"Training accuracy: {preference_accuracy:.2f}%")
    
    # Test
    ytest = estimator.mean(xtest)
    y_ground_truth = benchmark_f.eval_noiseless(xtest)
    
    # Calculate test accuracy using random pairs from test set
    test_pair_indices = torch.randint(n, size=(M, K))  # Using same M as training
    test_pairs = xtest[test_pair_indices].reshape(-1, 1)
    test_f_values = benchmark_f.eval_noiseless(test_pairs).reshape(M, K)
    
    # Generate ground truth preferences for test pairs
    test_probs = torch.softmax(test_f_values, dim=1)
    test_samples = torch.multinomial(test_probs, num_samples=1).squeeze()
    test_y = torch.zeros((M, K))
    test_y[torch.arange(M), test_samples] = 1
    
    # Get predictions for test pairs
    test_pair_values = estimator.mean(xtest[test_pair_indices].reshape(-1, 1)).reshape(M, K)
    test_predicted_preferences = torch.zeros((M, K))
    test_predicted_preferences[torch.arange(M), torch.argmax(test_pair_values, dim=1)] = 1
    test_accuracy = (torch.sum(test_predicted_preferences == test_y).item() / (M * K)) * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(xtest.numpy(), y_ground_truth.numpy(), 'b--', label='True function')
    plt.plot(xtest.numpy(), ytest.numpy(), 'r--', label='Estimated function')
    plt.legend()
    plt.show()
