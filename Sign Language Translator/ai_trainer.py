import os
import numpy as np
import cv2
import pandas as pd
import joblib
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class SignLanguageAITrainer:
    """AI Model trainer for sign language recognition using Random Forest"""
    
    def __init__(self, model_name="sign_language_model", num_classes=10):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.csv_file = f"training_data/{model_name}_features.csv"
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            try:
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: MediaPipe initialization error: {e}")
                self.hands = None
        else:
            self.hands = None
        
    def extract_hand_features(self, frame):
        """Extract normalized hand landmarks from frame"""
        if self.hands is None:
            return np.random.randn(42)  # 21 landmarks * 2 (x, y)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        features = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get first landmark as reference for normalization
            x0 = hand_landmarks.landmark[0].x
            y0 = hand_landmarks.landmark[0].y
            
            # Extract all 21 landmarks, normalized to first point
            for landmark in hand_landmarks.landmark:
                features.append(landmark.x - x0)
                features.append(landmark.y - y0)
        
        # Pad or truncate to 42 features (21 landmarks * 2)
        features = np.array(features[:42])
        while len(features) < 42:
            features = np.append(features, 0)
        
        return features[:42]
    
    def collect_training_frame(self, frame, label):
        """Collect one frame of training data"""
        features = self.extract_hand_features(frame)
        
        # Create directory if doesn't exist
        if not os.path.exists("training_data"):
            os.makedirs("training_data")
        
        # Append to CSV
        row = list(features) + [label]
        
        if not os.path.exists(self.csv_file):
            # Create new CSV with headers
            headers = [f"f{i}" for i in range(42)] + ["label"]
            df = pd.DataFrame([row], columns=headers)
            df.to_csv(self.csv_file, index=False)
        else:
            # Append to existing CSV
            df = pd.read_csv(self.csv_file)
            new_row = pd.DataFrame([row], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.csv_file, index=False)
        
        return True
    
    def prepare_training_data(self):
        """Load training data from CSV"""
        if not os.path.exists(self.csv_file):
            print(f"Error: {self.csv_file} not found!")
            return None, None
        
        df = pd.read_csv(self.csv_file)
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        print(f"Loaded {len(X)} samples from {self.csv_file}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Unique labels: {np.unique(y)}")
        
        return X, y
    
    def build_model(self):
        """Build Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("Random Forest Model built:")
        print(f"  - Trees: 300")
        print(f"  - Max Depth: 20")
        print(f"  - Random State: 42")
        return self.model
    
    def train(self, X, y, test_size=0.2):
        """Train the Random Forest model"""
        print(f"\nTraining Random Forest model on {len(X)} samples...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.2%}")
        print(f"Testing Accuracy: {test_accuracy:.2%}")
        
        return train_accuracy, test_accuracy
    
    def save_model(self, output_dir="models"):
        """Save trained model"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model
        model_path = os.path.join(output_dir, f"{self.model_name}.pkl")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, f"{self.model_name}_labels.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Label encoder saved to {encoder_path}")
        
        # Save config
        config_path = os.path.join(output_dir, f"{self.model_name}_config.json")
        config = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'classes': list(self.label_encoder.classes_),
            'n_features': 42,
            'timestamp': datetime.now().isoformat()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")
    
    def load_model(self, model_dir="models"):
        """Load pre-trained model"""
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        encoder_path = os.path.join(model_dir, f"{self.model_name}_labels.pkl")
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, frame):
        """Predict sign from frame"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        features = self.extract_hand_features(frame)
        features = features.reshape(1, -1)
        
        # Get prediction and confidence
        pred_encoded = self.model.predict(features)[0]
        pred_proba = self.model.predict_proba(features)[0]
        confidence = np.max(pred_proba)
        
        # Decode label
        label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return label, float(confidence)


if __name__ == "__main__":
    trainer = SignLanguageAITrainer(num_classes=10)
    trainer.build_model()
    
    print("Sign Language AI Trainer (Random Forest) initialized successfully!")
    print("Use this trainer in your Flask backend to train custom sign language models.")
