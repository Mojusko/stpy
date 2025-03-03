import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from stpy.estimator import Estimator
import torch.nn.init as init
import torch.func as func
from stpy.regression.nn.mlp import MLPRegressor
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from stpy.probability.likelihood import MultinomialLikelihood
class CNNForMNIST(MLPRegressor):
    def __init__(self,
                 layer_sizes,     # not really used for the CNN structure, but needed for parent's signature
                 likelihood,
                 regularizer,
                 activation=nn.ReLU,
                 output_activation=None,
                 learning_rate=0.0001,
                 verbose=True,
                 epochs=10,
                 batch_size=32,
                 device='cpu'):
        super(CNNForMNIST, self).__init__(layer_sizes,
                                          likelihood,
                                          regularizer,
                                          activation=activation,
                                          output_activation=output_activation,
                                          learning_rate=learning_rate,
                                          verbose=verbose,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          device=device)

        # ----------------------------------------------------------------
        # OVERRIDE the parent "model" with a simple 2-conv-layer architecture for MNIST
        # ----------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(32 * 24 * 24, 10)  # 10 classes for MNIST

        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        self.fc    = self.fc.to(device)

        # Weight init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # Re-initialize the optimizer so it includes our new conv and fc parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        """
        Override forward to do conv -> relu -> conv -> relu -> flatten -> fc.
        x should be (batch_size, 1, 28, 28).
        """
        x = x.to(self.device)
        x = nn.functional.relu(self.conv1(x))  # out: (batch_size, 16, 26, 26)
        x = nn.functional.relu(self.conv2(x))  # out: (batch_size, 32, 24, 24)
        x = x.view(x.size(0), -1)             # flatten: (batch_size, 32*24*24)
        logits = self.fc(x)                   # (batch_size, 10)
        return logits

    def fit(self, train_loader):
        """
        Custom fit loop for classification on MNIST using cross-entropy.
        We'll do basically the same steps as the parent but for clarity, we restate them.
        """
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.forward(batch_x)
                # We assume likelihood is cross-entropy
                loss = self.likelihood(logits, batch_y) + self.regularizer(self)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}")


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Optionally normalize images.
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # -------------------
    # 2) Build model
    # -------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    likelihood = MultinomialLikelihood(classes = 10)  # from above
    regularizer = NoRegularizer()  # from above

    # layer_sizes is not used for the CNN structure, but we pass something to satisfy MLPRegressor
    layer_sizes = [784, 10]

    model = CNNForMNIST(layer_sizes=layer_sizes,
                        likelihood=likelihood,
                        regularizer=regularizer,
                        activation=nn.ReLU,
                        learning_rate=1e-3,
                        epochs=5,
                        batch_size=64,
                        verbose=True,
                        device=device)

    # -------------------
    # 3) Train model
    # -------------------
    model.fit(train_loader)

    # After training, you could evaluate on a test set, etc.
    # Example: test on a single batch
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            logits = model.forward(x_test)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == y_test).item()
            total += y_test.size(0)
    print(f"Test Accuracy: {correct / total:.2%}")