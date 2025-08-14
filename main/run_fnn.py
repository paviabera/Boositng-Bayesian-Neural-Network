import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.fnn import SimpleFNN
from data.preprocess_classification import load_classification_data
from utils.train_eval import train_model, evaluate_model
from data.preprocess_regression import generate_branin_data

# Classification DataSeta
# dataset = "DataSets/diabetes.csv"
# target_col = "Outcome"

# dataset = "DataSets/cancer.csv"
# target_col = "diagnosis"

# dataset = "DataSets/hepatitis.csv"
# target_col = "out_class"

# dataset = "DataSets/heart.csv"
# target_col = "target"

# dataset = "DataSets/heart_statlog_cleveland_hungary_final.csv"
# target_col = "target"

# X_train, X_test, y_train, y_test, input_dim, output_dim = load_classification_data(dataset, target_col)

# Regression generated dataSets
# X_train, X_test, y_train, y_test, input_dim, output_dim = generate_synthetic_regression_data()

# Regression Branin Data
X_train, X_test, y_train, y_test, input_dim, output_dim = generate_branin_data()

model = SimpleFNN(input_dim=input_dim, hidden_dim=16, output_dim=output_dim)
train_model(model, X_train, y_train, epochs=100, classification=False)
evaluate_model(model, X_test, y_test, classification=False)
