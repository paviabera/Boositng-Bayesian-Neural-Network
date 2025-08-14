import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def load_dataset(file_path, test_size=0.3, random_state=42):
    df = pd.read_csv(file_path)
    name = file_path.lower()

    # 🔍 Dataset-specific logic
    if "cancer" in name:
        target = "diagnosis"
        df[target] = LabelEncoder().fit_transform(df[target])  # M/B → 1/0
        X = df.iloc[:, 2:].values
        y = df[target].values

    elif "diabetes" in name:
        target = "Outcome"
        X = df.iloc[:, :-1].values
        y = df[target].values

    elif "hepatitis" in name:
        df.replace("?", np.nan, inplace=True)
        df["out_class"] = df["out_class"].replace({2: 1, 1: 0})
        y = df["out_class"].astype(int).values
        binary_cols = [
            "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia",
            "liver_big", "liver_firm", "spleen_palable", "spiders", "ascites",
            "varices", "histology"
        ]
        df[binary_cols] = df[binary_cols].replace({1: 0, 2: 1})
        df.fillna(df.mean(), inplace=True)
        numerical_cols = ["age", "bilirubin", "alk_phosphate", "sgot", "albumin", "protime"]
        df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])
        X = df.iloc[:, 1:].values

    elif "heart" in name:
        target = "target"
        y = df[target].values
        X = df.drop(columns=[target])
        X = pd.get_dummies(X).values

    else:
        raise ValueError(f"Dataset '{file_path}' is not supported.")

    # Standardize
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        X.shape[1],
        len(np.unique(y))  # Usually 2 for binary classification
    )

# data/preprocess_classification.py
# data/preprocess_classification.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_classification_data(csv_path, test_size=0.3, random_state=42, device="cpu"):
    df = pd.read_csv(csv_path)
    
    # Assumes last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels if they are not numeric
    if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    output_dim = len(set(y))

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_dim, output_dim
