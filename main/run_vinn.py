import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vinn import VariationalNN
from pyro.infer.autoguide import AutoDiagonalNormal
from data.preprocess_regression import generate_branin_data
from utils.train_eval_vinn import train_vinn, evaluate_vinn

from data.preprocess_classification import load_dataset

from config.device_config import device
import torch
import argparse

print("Using device:", device)

if args.dataset == "branin":
    from data.preprocess_regression import generate_branin_data
    X_train, X_test, y_train, y_test, input_dim, output_dim = generate_branin_data()
    classification = False
else:
    file_path = os.path.join("DataSets", f"{args.dataset}.csv")
    X_train, X_test, y_train, y_test, input_dim, output_dim = load_dataset(file_path)
    classification = True

model = VariationalNN(input_dim, hidden_dim=16, output_dim=output_dim).to(device)
guide = AutoDiagonalNormal(model)

train_vinn(model, guide, X_train, y_train, classification=False, epochs=1000)
evaluate_vinn(model, guide, X_test, y_test, classification=False)
