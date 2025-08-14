import pyro
import pyro.infer
import pyro.optim
import torch
from config.device_config import device

def train_vinn(model, guide, X_train, y_train, classification=True, lr=0.01, epochs=1000):
    pyro.clear_param_store()
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)

    loss_fn = pyro.infer.Trace_ELBO()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=loss_fn)

    for epoch in range(epochs):
        loss = svi.step(X_train, y_train)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, ELBO Loss: {loss:.4f}")
            

def evaluate_vinn(model, guide, X_test, y_test, classification=True, num_samples=100):
    from pyro.infer import Predictive
    model.eval()
    guide.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    preds = predictive(X_test)
    y_pred_samples = preds["prediction"]


    if classification:
        probs = torch.mean(torch.softmax(y_pred_samples, dim=-1), dim=0)
        pred_class = torch.argmax(probs, dim=1)
        accuracy = (pred_class == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy*100:.2f}%")
    else:
        pred_mean = torch.mean(y_pred_samples, dim=0)
        mse = ((pred_mean - y_test) ** 2).mean().item()
        print(f"Test MSE: {mse:.4f}")
