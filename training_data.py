import csv
import os
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier


def load_model_state(model_path):
    stored = joblib.load(model_path)
    if isinstance(stored, dict) and "model" in stored and "disabled_motion" in stored:
        return stored["model"], set(stored["disabled_motion"])
    return stored, set()


def save_model_state(model_path, model, disabled_motion):
    directory = os.path.dirname(os.path.abspath(model_path))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary_path = temporary.name
        joblib.dump({"model": model, "disabled_motion": sorted(disabled_motion)}, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def fit_model(data):
    if data.empty:
        return None
    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=20)
    model.fit(data.iloc[:, :-1], data.iloc[:, -1])
    return model


def delete_label_samples(csv_path, label):
    normalized_label = label.strip().casefold()
    if not normalized_label:
        raise ValueError("A label is required.")

    directory = os.path.dirname(os.path.abspath(csv_path))
    temporary_path = None
    removed = 0

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=directory,
                delete=False,
            ) as destination:
                temporary_path = destination.name
                reader = csv.reader(source)
                writer = csv.writer(destination)

                for row in reader:
                    if row and row[-1].strip().casefold() == normalized_label:
                        removed += 1
                    else:
                        writer.writerow(row)

        os.replace(temporary_path, csv_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise

    return removed
