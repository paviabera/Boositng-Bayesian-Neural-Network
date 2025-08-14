import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import argparse
import os
from data.preprocess_classification import load_classification_data
from utils.train_eval_bvinn import train_bvinn, evaluate_bvinn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bimodal_classification_dataset.csv")
    parser.add_argument("--loss_type", default="elbo", help="Choose 'elbo' or 'relbo'")
    args = parser.parse_args()

    file_path = os.path.join("DataSets", args.dataset)
    X_train, X_test, y_train, y_test, input_dim, output_dim = load_classification_data(file_path)

    model, guide = train_bvinn(X_train, y_train, input_dim, output_dim, loss_type=args.loss_type)
    evaluate_bvinn(model, guide, X_test, y_test)
