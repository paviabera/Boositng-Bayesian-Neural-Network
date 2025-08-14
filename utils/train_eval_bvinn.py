import torch
import pyro
import pyro.optim as optim
from pyro.infer import Trace_ELBO, TraceMeanField_ELBO, Trace_RELBO
from pyro.infer import SVI

from models.bvinn import BayesianNN

def train_bvinn(X_train, y_train, input_dim, output_dim, hidden_dim=10, loss_type="elbo", n_iterations=1000, lr=0.01):
    pyro.clear_param_store()
    model = BayesianNN(input_dim, hidden_dim, output_dim)
    
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optimizer = optim.Adam({"lr": lr})

    if loss_type.lower() == "relbo":
        loss = Trace_RELBO()
    else:
        loss = Trace_ELBO()
        
    svi = SVI(model, guide, optimizer, loss=loss)

    for step in range(n_iterations):
        loss_val = svi.step(X_train, y_train)
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss_val:.4f}")
    
    return model, guide

def evaluate_bvinn(model, guide, X_test, y_test, num_samples=100):
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=num_samples)
    samples = predictive(X_test)
    pred_mean = samples["obs"].mode(dim=0)[0]
    
    accuracy = (pred_mean == y_test).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")
