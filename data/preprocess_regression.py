import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt.benchmarks import branin


def generate_synthetic_regression_data(n_samples=300, noise_std=2.0, test_size=0.3, random_state=42):
    # Sample from uniform grid
    X = np.random.uniform(-5, 10, size=(n_samples, 2))
    y = (
        (X[:, 1] - 5.1 * X[:, 0]**2 / (4 * np.pi**2) + 5 * X[:, 0] / np.pi - 6)**2
        + 10 * (1 - 1/(8 * np.pi)) * np.cos(X[:, 0]) + 10
    )  # Branin function
    
    # Add Gaussian noise
    y += np.random.normal(scale=noise_std, size=y.shape)

    # Standardize inputs
    X = StandardScaler().fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        X.shape[1],  # input_dim = 2
        1,           # output_dim = 1 (regression)
    )

def generate_branin_data(n_samples=300, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    
    # Branin domain: x ∈ [-5, 10], y ∈ [0, 15]
    X = np.random.uniform([ -5,  0], [10, 15], size=(n_samples, 2))
    y = np.array([branin(x) for x in X])

    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        X.shape[1],  # input_dim = 2
        1            # output_dim = 1
    )
