import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from training_data import load_model_state, save_model_state

print("Loading training data...")
data = pd.read_csv("sign_data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Labels: {y.unique()}")

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest model (this takes ~10 seconds)...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=20
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"\n✅ Training Complete!")
print(f"Accuracy: {accuracy:.2%}")

print("\nSaving model...")
model.fit(X, y)
model_path = "models/sign_language_model.pkl"
disabled_motion = set()
if os.path.exists(model_path):
    _, disabled_motion = load_model_state(model_path)
labels = {str(label).strip().upper() for label in y}
save_model_state(model_path, model, disabled_motion - labels)
print("✅ Model saved to models/sign_language_model.pkl")
