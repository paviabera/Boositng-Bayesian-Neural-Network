import torch
import time
from config.device_config import device

def train_model(model, X_train, y_train, epochs=100, lr=1e-3, classification=True):
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)

    criterion = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def evaluate_model(model, X_test, y_test, classification=True):
    model.eval()
    model.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        outputs = model(X_test)
        if classification:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).float().mean().item()
            print(f"Test Accuracy: {accuracy*100:.2f}%")
        else:
            mse = torch.nn.functional.mse_loss(outputs.squeeze(), y_test.float())
            print(f"Test MSE: {mse.item():.4f}")
