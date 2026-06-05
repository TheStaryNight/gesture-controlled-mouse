import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

CLASSES = {
    0: "Move (Hover)",
    1: "Left Click",
    2: "Right Click",
    3: "Scroll Mode"
}

def load_data():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Dataset '{CSV_FILE}' not found. Please run preprocess_dataset.py first.")
        return None, None

    data = []
    labels = []
    with open(CSV_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            # Skip header row if present
            if parts[0].startswith("lm_"):
                continue
            if len(parts) == 43:  # 42 features + 1 label
                try:
                    data.append([float(x) for x in parts[:-1]])
                    labels.append(int(parts[-1]))
                except ValueError:
                    pass  # Ignore malformed rows

    return np.array(data), np.array(labels)

def main():
    X, y = load_data()
    if X is None or len(X) == 0:
        return

    print(f"Loaded dataset with {len(X)} samples.")
    for label_id, label_name in CLASSES.items():
        count = np.sum(y == label_id)
        print(f"  Class {label_id} ({label_name}): {count} samples")

    # Ensure we have data for all classes to train
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Error: You need at least 2 different gesture classes to train the AI model.")
        return

    # Train/Test Split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n--- Training Model 1: Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Validation Accuracy: {rf_acc:.4f}")

    print("\n--- Training Model 2: Multi-Layer Perceptron (Neural Network) ---")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    mlp_clf.fit(X_train, y_train)
    mlp_pred = mlp_clf.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    print(f"MLP Neural Network Validation Accuracy: {mlp_acc:.4f}")

    # Compare models and select best
    print("\n==================================================")
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"Neural Network (MLP) Accuracy: {mlp_acc*100:.2f}%")
    
    if rf_acc >= mlp_acc:
        best_model = rf_clf
        best_name = "Random Forest Classifier"
        best_acc = rf_acc
        best_pred = rf_pred
    else:
        best_model = mlp_clf
        best_name = "MLP Neural Network"
        best_acc = mlp_acc
        best_pred = mlp_pred
        
    print(f"==> Selected Best Model: {best_name} ({best_acc*100:.2f}% Accuracy)")
    print("==================================================")

    # Print detailed classification report for the best model
    class_names = [CLASSES[i] for i in sorted(unique_classes)]
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, best_pred, target_names=class_names))

    # Save model to disk
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved successfully to '{MODEL_FILE}'.")

if __name__ == "__main__":
    main()
